import torch
from torch.fx import Interpreter
import torch._dynamo as dynamo
from Opara import OperatorLauncher
from Opara import StreamAllocator
from Opara import TimeConstrainedScheduler
from Opara import TimeConstrainedResourceAwareScheduler
from torch._functorch.partitioners import draw_graph
import os
path = os.path.abspath(os.path.dirname(__file__))
output_file_path = path + '/profile_result/output.txt'
output_file = open(output_file_path, "w")

class Scheduler(Interpreter):
    def run_node(self, n):
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        # print(n)
        # 等待前置事件 (Wait Events)
        # 如果当前节点需要等待某些前驱节点完成（比如跨流依赖），在这里阻塞当前流
        for event in n.event_to_wait:
            # print(n.name, n.stream)
            n.stream.wait_event(event)

        # 在节点所属 stream 上执行，并确保退出时恢复到之前的 current stream。
        # 这对 CUDA Graph capture 场景尤为重要：异常情况下也能避免
        # "Capture must end on the same stream it began on"。
        with torch.cuda.stream(n.stream):
            # 提取当前节点n所需的位置参数和关键字参数，即真实的tensor数据
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)

            # 实际执行算子n，并将结果保存到env
            self.env[n] = getattr(self, n.op)(n.target, args, kwargs)
        
        # n.event.record(n.stream)

        # 记录事件 (Record Event)
        # 检查当前节点的后继节点，如果后继节点在不同的流上，则记录事件供其等待
        is_record = False
        #n.users是所有把节点n作为输入的后续节点
        for user in n.users:
            if n.stream != user.stream:
                if is_record is False:
                    n.event.record(n.stream)
                    is_record = True
        return self.env[n]
    
    def run(self, *args):
        """
        Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order
            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.
                This is a dict mapping `Node` to any value. This can be used, for example, to
                pre-populate results for certain `Nodes` so as to do only partial evaluation within
                the interpreter.
            enable_io_processing (bool): If true, we process the inputs and outputs with graph's process_inputs and
                process_outputs function first before using them.

        Returns:
            Any: The value returned from executing the Module
        """
        self.env = {}
        self.args_iter = iter(args)
        # Positional function args are consumed left-to-right byp
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        # print("run_node->len(graph.nodes):", len(self.module.graph.nodes))
        for node in self.module.graph.nodes:
            # print("run_node->node:", node)
            self.env[node] = self.run_node(node)

            if node.op == 'output':
                output_val = self.env[node]
                return output_val
            


def capturer(
    inputs,
    model,
    copy_outputs: bool = False,
    use_tcas: bool = False,
    use_tcas_ra: bool = False,
    tcas_epsilon: float = 1e-4,
    tcas_ra_depvalue_epsilon_abs: float = 1e-6,
    tcas_ra_depvalue_epsilon_rel: float = 1e-9,
):
    """
    将模型转换为可并行执行的 CUDA Graph
    
    Args:
        inputs: 模型输入
        model: PyTorch 模型
        copy_outputs: 是否复制输出
        use_tcas: 是否使用 DepValue 调度器重排 FX（仍走 StreamAllocator 贪心流分配）
        use_tcas_ra: 是否使用 TCAS-RA（DepValue 梯队内算存交错 + StreamAllocator）；与 use_tcas 互斥，优先 use_tcas_ra
        tcas_ra_depvalue_epsilon_abs / tcas_ra_depvalue_epsilon_rel: TCAS-RA 梯队阈值
    """
    assert isinstance(inputs, (list, tuple)), f"inputs is of type {type(inputs)} instead of list"

    # 跟随输入所在设备（尤其是多 GPU 场景）。
    # 约定：inputs 均为 Tensor 且位于同一张 GPU 上。
    input0 = inputs[0]
    device = input0.device if isinstance(input0, torch.Tensor) else torch.device('cuda')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # 准备静态输入 (Static Inputs)
    # CUDA Graph 需要固定的内存地址，所以我们创建一组全零的 Tensor 作为占位符
    static_inputs = [torch.zeros_like(x) for x in inputs]

    dynamo.reset()
    with torch.no_grad():
        # ====================== 【核心修复开始】完美适配新旧 PyTorch ======================
        try:
            # 【新版 PyTorch】API: explain(f)(*args)
            explainer = dynamo.explain(model)
            explain_output = explainer(*inputs)
            
            # 从 ExplainOutput 对象中提取字段
            graphs = getattr(explain_output, 'graphs', [])
            break_reasons = getattr(explain_output, 'break_reasons', [])
            explanation = getattr(explain_output, 'explanation', None)
            out_guards = None
            ops_per_graph = [len(g.graph.nodes) for g in graphs] if graphs else []
            explanation_verbose = None

        except (TypeError, AttributeError):
            # 【旧版 PyTorch】回退: explain(f, *args)
            explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose = dynamo.explain(model, *inputs)
        # ====================== 【核心修复结束】 ======================
        
    fx_module = graphs[0]
    # print(fx_module.graph, file=output_file)
    fx_module.to(device)
    model_class_name = model.__class__.__name__
    
    # 获取 Profile 信息和设备属性
    # 若使用 TCAS / TCAS-RA，不应用 Opara 的 launch 重排（由后续调度器写回 FX）
    apply_opara = not (use_tcas or use_tcas_ra)
    node_profiles, device_props = OperatorLauncher.recompile(
        model_class_name,
        fx_module,
        inputs,
        apply_opara_schedule=apply_opara,
    )

    # 选择调度算法（use_tcas_ra 优先于 use_tcas）
    if use_tcas_ra:
        print("[GraphCapturer] 使用 TCAS-RA (DepValue 梯队内算存交错 + StreamAllocator)")
        TimeConstrainedResourceAwareScheduler.assign_stream_with_tcas_ra(
            fx_module,
            node_profiles=node_profiles,
            device_props=device_props,
            depvalue_epsilon_abs=tcas_ra_depvalue_epsilon_abs,
            depvalue_epsilon_rel=tcas_ra_depvalue_epsilon_rel,
        )
        all_streams, all_events = StreamAllocator.assign_stream(fx_module.graph)
    elif use_tcas:
        # TimeConstrainedScheduler.assign_stream_with_tcas：DepValue 列表调度 + 写回 FX
        print("[GraphCapturer] 使用 DepValue 调度器 (TimeConstrainedScheduler)")
        TimeConstrainedScheduler.assign_stream_with_tcas(
            fx_module,
            node_profiles=node_profiles,
            epsilon=tcas_epsilon,
        )
        all_streams, all_events = StreamAllocator.assign_stream(fx_module.graph)
    else:
        print("[GraphCapturer] 使用原始 Opara 调度器")
        all_streams, all_events = StreamAllocator.assign_stream(fx_module.graph)

    # 收集结构性元数据（用于基准/论文分析，不影响执行）
    try:
        stream_id_map = {id(s): i for i, s in enumerate(all_streams)}
        per_stream_node_counts = [0 for _ in range(len(all_streams))]
        cross_stream_wait_events = 0

        for n in fx_module.graph.nodes:
            sid = stream_id_map.get(id(getattr(n, 'stream', None)), -1)
            if 0 <= sid < len(per_stream_node_counts):
                per_stream_node_counts[sid] += 1
            cross_stream_wait_events += len(getattr(n, 'event_to_wait', []) or [])

        graph_node_names_head = [n.name for n in fx_module.graph.nodes][:30]

        if use_tcas_ra:
            algo_name = 'TCAS-RA'
        elif use_tcas:
            algo_name = 'DepValue'
        else:
            algo_name = 'Opara'

        _bench_meta = {
            'algorithm': algo_name,
            'num_nodes': len(list(fx_module.graph.nodes)),
            'num_streams': len(all_streams),
            'per_stream_node_counts': per_stream_node_counts,
            'cross_stream_wait_events': int(cross_stream_wait_events),
            'graph_nodes_head': graph_node_names_head,
        }
        if use_tcas:
            _bench_meta['tcas_epsilon'] = float(tcas_epsilon)
        if use_tcas_ra:
            _bench_meta['tcas_ra_depvalue_epsilon_abs'] = float(tcas_ra_depvalue_epsilon_abs)
            _bench_meta['tcas_ra_depvalue_epsilon_rel'] = float(tcas_ra_depvalue_epsilon_rel)
    except Exception:
        _bench_meta = {
            'algorithm': (
                'TCAS-RA' if use_tcas_ra else ('DepValue' if use_tcas else 'Opara')
            )
        }

    # 准备 CUDA Graph 录制环境
    all_events = [torch.cuda.Event(enable_timing=False) for _ in range(len(all_streams))]
    first_stream = all_streams[0]
    first_event = all_events[0]
    interpreter = Scheduler(fx_module)

    # with torch.autocast(device_type='cuda', dtype=torch.float16):

    # Warmup (预热)
    # 在录制前先跑几遍，确保 CUDA Context 初始化完毕
    with torch.no_grad():
        for i in range(3):
            interpreter.run(*inputs)
    with torch.no_grad():
        # 录制 CUDA Graph
        g = torch.cuda.CUDAGraph()

        # 重要：capture 过程中会切换 stream；无论成功或异常，都确保在退出 capture
        # 作用域前恢复到 first_stream。
        with torch.cuda.graph(g, stream=first_stream):
            try:
                # 扇出 (Fan-out) 同步：确保所有子流都等待主流开始
                first_event.record(first_stream)

                for i, stream in enumerate(all_streams):
                    if i > 0:
                        stream.wait_event(first_event)

                # 执行模型：调用 interpreter.run 并行执行所有算子
                static_outputs = interpreter.run(*static_inputs)

                # 扇入 (Fan-in) 同步：确保主流等待所有子流结束
                torch.cuda.set_stream(first_stream)
                for i, event in enumerate(all_events):
                    if i > 0:
                        event.record(all_streams[i])
                for i, event in enumerate(all_events):
                    if i > 0:
                        first_stream.wait_event(event)
            finally:
                torch.cuda.set_stream(first_stream)

        torch.cuda.synchronize()

        if not isinstance(static_outputs, (list, tuple)):
            static_outputs = (static_outputs,)

    # 定义返回给用户的执行函数
    def run(*new_inputs):
        assert isinstance(new_inputs, (list, tuple)), f"inputs is of type {type(new_inputs)} instead of list"
        assert len(static_inputs) == len(new_inputs), f"{len(static_inputs)} == {len(new_inputs)}"
        # 数据拷贝：将新输入拷贝到静态显存地址
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)  # cuda graph can only read data from the same address
        with torch.no_grad():
            # 重放 Graph，此时cpu可以直接向gpu发送整个graph，无需cpu逐个发射算子
            g.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        else:
            return static_outputs

    # 将元数据挂到返回的闭包上，便于外部脚本读取（不改变返回值类型）
    try:
        setattr(run, '_opara_meta', _bench_meta)
    except Exception:
        pass

    return run
