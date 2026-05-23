from functools import reduce
from operator import mul
from Opara import ModelProfiler

from collections import deque
import os
path = os.path.abspath(os.path.dirname(__file__))
output_file_path = path + '/profile_result/output.txt'
output_file = open(output_file_path, "w")

#这一部分是决定————交替执行计算密集型队列与内存密集型队列中的算子；
#用is_mem_access_intensive 来区分每个算子属性
#每一个node是一个算子，一个算子包含1个或者多个kernel函数，kernel函数是gpu上执行的
#比如relu对应relu_kernel,一个attention算子包含多个kernel，gemm+softmax+gemm三个kernel组成

#用pop_from_queue来选择每个队列中显存占用最小的算子，先填入小算子，充分利用碎片资源
#接下来就是交错调度，用flag强制交替两个队列，以达到显存和sm都忙碌的目的
#最终得到的result就是算子的发射顺序，这里是需要和stream分配进行结合，以形成最优的CUDA Graph

# 典型 GEMM/Conv/Attention：先判为计算密集，避免 addmm 因含 "add" 被误判（与 TCAS-RA 共用）
_COMPUTE_HEAVY_SUBSTRINGS = (
    "addmm",
    "bmm",
    "matmul",
    "linear",
    "conv2d",
    "conv1d",
    "conv3d",
    "convolution",
    "cudnn_convolution",
    "_convolution",
    "einsum",
    "scaled_dot_product",
    "sdpa",
    "flash_attn",
    "grouped_gemm",
    "gemm",
)

# 元素级/规约/布局/池化/归一化/Embedding 等（ResNet/BERT/Inception/DeepFM 常见 FX 名）
_MEMORY_INTENSIVE_SUBSTRINGS = (
    "add",
    "cast",
    "ceil",
    "clip",
    "concat",
    "exp",
    "floor",
    "log",
    "gelu",
    "neg",
    "pow",
    "reciprocal",
    "relu",
    "sigmoid",
    "slice",
    "sqrt",
    "sub",
    "tanh",
    "transpose",
    "unsqueeze",
    "view",
    "avg_pool",
    "reshape",
    "max_pool",
    "adaptive_avg_pool",
    "adaptive_max_pool",
    "permute",
    "flatten",
    "dropout",
    "batch_norm",
    "layer_norm",
    "instance_norm",
    "contiguous",
    "ones",
    "to",
    "softmax",
    "native_layer_norm",
    "rms_norm",
    "masked_fill",
    "embedding",
    "embedding_bag",
    "index_select",
    "gather",
    "where",
    "silu",
    "hardswish",
    "pad",
    "clone",
    "split",
    "chunk",
    "repeat",
    "expand",
    "cumsum",
    "one_hot",
    "arange",
)


def is_mem_access_intensive_by_name(name: str) -> bool:
    """根据 FX 节点名判断偏访存/轻量（True）或偏计算（False）；与 TimeConstrainedResourceAwareScheduler 一致。"""
    lower = (name or "").lower()
    if lower == "mm":
        return False
    if any(s in lower for s in _COMPUTE_HEAVY_SUBSTRINGS):
        return False
    if any(s in lower for s in _MEMORY_INTENSIVE_SUBSTRINGS):
        return True
    return False


def is_mem_access_intensive_node(node) -> bool:
    if node is None:
        return False
    return is_mem_access_intensive_by_name(getattr(node, "name", ""))


def launch(nodes, result, in_degree, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock):
    def pop_from_queue(q):
        ret_node_name = q[0]

        min_metric = 2
        for node_name in q:
            if len(nodes[node_name].info) > 0:
                achieved_occupancy = nodes[node_name].info[0]["args"]["est. achieved occupancy %"]
                blocksPerSM = nodes[node_name].info[0]["args"]["blocks per SM"]
                shared_memory = nodes[node_name].info[0]["args"]["shared memory"] / sharedMemPerBlock
                thread_num = reduce(mul, nodes[node_name].info[0]["args"]["block"]) / maxThreadsPerBlock
                registers_num = thread_num * nodes[node_name].info[0]["args"]["registers per thread"] / regsPerBlock
                request = [shared_memory, thread_num, registers_num]
                # metric = max(shared_memory, max(thread_num, registers_num))
                # metric = thread_num
                # metric = achieved_occupancy
                metric = shared_memory
                # if metric == min_metric:
                #     print("same metric:", node_name, ret_node_name)
                if metric < min_metric:
                    min_metric = metric
                    ret_node_name = node_name
        q.remove(ret_node_name)

        return ret_node_name
    
    memory_queue = deque()
    not_memory_queue = deque()
    for node_name, degree in in_degree.items():
        if degree == 0:
            if is_mem_access_intensive_node(nodes.get(node_name)):
                memory_queue.append(node_name)
            else:
                not_memory_queue.append(node_name)
    flag = True
    while memory_queue or not_memory_queue:

        flag = not flag
        if memory_queue and not_memory_queue:
            if flag:
                q = memory_queue
            else:
                q = not_memory_queue
        else:
            if memory_queue:
                q = memory_queue
            else:
                q = not_memory_queue
        cur_node_name = pop_from_queue(q)
        result.append(cur_node_name)

        for succ_node in nodes.get(cur_node_name).users:
            in_degree[succ_node.name] -= 1
            if in_degree[succ_node.name] == 0:
                if is_mem_access_intensive_node(nodes.get(succ_node.name)):
                    memory_queue.append(succ_node.name)
                else:
                    not_memory_queue.append(succ_node.name)
    return result


def launch_with_cp_tiebreak(
    nodes,
    result,
    in_degree,
    sharedMemPerBlock,
    regsPerBlock,
    maxThreadsPerBlock,
    slack_by_name,
    critical_by_name,
    metric_eps: float = 1e-9,
):
    """
    与 launch() 相同的双队列交替 + 队内 shared 最小优先；
    仅当多个候选的 metric（与 launch 一致：首个 kernel 的 shared/sharedMemPerBlock）同为最小时，
    用关键路径与 slack 做 tie-break：优先 is_critical，再优先更小 slack，最后按名字稳定序。
    slack_by_name / critical_by_name：节点名字符串 -> float / bool。
    """

    def pop_from_queue(q):
        metrics = []
        for node_name in q:
            if len(nodes[node_name].info) > 0:
                m = nodes[node_name].info[0]["args"]["shared memory"] / sharedMemPerBlock
                metrics.append((node_name, m))
        if not metrics:
            pick = q[0]
            q.remove(pick)
            return pick

        min_m = min(m for _, m in metrics)
        tied = [n for n, m in metrics if abs(m - min_m) <= metric_eps]
        if len(tied) == 1:
            pick = tied[0]
        else:
            pick = sorted(
                tied,
                key=lambda n: (
                    0 if critical_by_name.get(n, False) else 1,
                    slack_by_name.get(n, float("inf")),
                    n,
                ),
            )[0]
        q.remove(pick)
        return pick

    memory_queue = deque()
    not_memory_queue = deque()
    for node_name, degree in in_degree.items():
        if degree == 0:
            if is_mem_access_intensive_node(nodes.get(node_name)):
                memory_queue.append(node_name)
            else:
                not_memory_queue.append(node_name)
    flag = True
    while memory_queue or not_memory_queue:
        flag = not flag
        if memory_queue and not_memory_queue:
            if flag:
                q = memory_queue
            else:
                q = not_memory_queue
        else:
            if memory_queue:
                q = memory_queue
            else:
                q = not_memory_queue
        cur_node_name = pop_from_queue(q)
        result.append(cur_node_name)

        for succ_node in nodes.get(cur_node_name).users:
            in_degree[succ_node.name] -= 1
            if in_degree[succ_node.name] == 0:
                if is_mem_access_intensive_node(nodes.get(succ_node.name)):
                    memory_queue.append(succ_node.name)
                else:
                    not_memory_queue.append(succ_node.name)
    return result


def get_topo_with_cp_tiebreak(
    fx_nodes,
    sharedMemPerBlock,
    regsPerBlock,
    maxThreadsPerBlock,
    slack_by_name,
    critical_by_name,
):
    nodes = {node.name: node for node in fx_nodes}
    in_degree = {node.name: 0 for node in nodes.values()}
    for node in nodes.values():
        for input_node in node.all_input_nodes:
            in_degree[node.name] += 1
    result = []
    result = launch_with_cp_tiebreak(
        nodes,
        result,
        in_degree,
        sharedMemPerBlock,
        regsPerBlock,
        maxThreadsPerBlock,
        slack_by_name,
        critical_by_name,
    )
    return result, nodes


#解析.json文件，提取算子的运行时数据，方便上面的代码做launch

import json
import os

def get_resource_from_json(path):
    with open(path) as f:
        data = json.load(f)

    step_num = 0
    for event in data["traceEvents"]:
        if "torch/fx/interpreter.py(97): run" in event["name"] and "run_node" not in event["name"]:
            step_num += 1
    # 新版 PyTorch / 不同 trace 格式下可能匹配不到 step，避免 // 0
    if step_num == 0:
        step_num = 1

    # 获取run_node事件、kernel_launch事件、kernel事件
    run_node_events = []
    kernel_launch_events = []
    kernel_events = []
    for event in data["traceEvents"]:
        if "run_node" in event["name"]:
            run_node_events.append(event)

        if event["name"] == "cudaLaunchKernel":
            kernel_launch_events.append(event)

        if event.get("cat", "None") == "kernel":
            kernel_events.append(event)


    # 计算获取一个step中的run_node事件、kernel_launch事件、kernel事件
    one_step_range_of_node = len(run_node_events) // step_num
    one_step_range_of_kernel_launch = len(kernel_launch_events) // step_num
    one_step_range_of_kernel = len(kernel_events) // step_num

    #只取最后一个step，跳过前期可能存在的warmup阶段，gpu运行更稳定
    start = step_num - 1
    end = step_num
    run_node_events = run_node_events[start*one_step_range_of_node:end*one_step_range_of_node]
    kernel_launch_events = kernel_launch_events[start*one_step_range_of_kernel_launch:end*one_step_range_of_kernel_launch]
    kernel_events = kernel_events[start*one_step_range_of_kernel:end*one_step_range_of_kernel]


    # 根据时间轴范围获取由node事件触发的kernel_launch事件
    node2kernels = []
    kernel_num = 0
    for i, node_event in enumerate(run_node_events):
        node2kernels.append([])
        for j, kernel_launch_event in enumerate(kernel_launch_events):
            if node_event["ts"] <= kernel_launch_event["ts"] and node_event["ts"] + node_event["dur"] >= kernel_launch_event["ts"]:
                node2kernels[i].append(kernel_events[j])
                kernel_num += 1

    # print("kernel_num", kernel_num)

    max_block_nums = []
    sum_time = 0
    for i, kernel_events in enumerate(node2kernels):
        
        max_block_size = 4096
        for kernel_event in kernel_events:
            sum_time += kernel_event["dur"]

            #计算当前kernel的block包含的总线程数
            cur_block_size = kernel_event["args"]["block"][0] * kernel_event["args"]["block"][1] * kernel_event["args"]["block"][2]

            #找该Node的所有kernel中线程数最少的那个？？？ 如果改成max_block_size初始设为0，max_block_size = max(max_block_size, cur_blcok_size),这样可以记录Node里kernel的最大线程数
            max_block_size = min(max_block_size, cur_block_size)
        #如果该Node未更新过，或者该Node没有对应的kernel，则置0
        if max_block_size == 4096:
            max_block_size = 0
        max_block_nums.append(max_block_size)

    #计算加权总占用率
    est_achieved_occupancy = 0
    for i, kernel_events in enumerate(node2kernels):
        for kernel_event in kernel_events:
            dur = kernel_event["dur"]
            est_achieved_occupancy += kernel_event["args"]["est. achieved occupancy %"] * dur
    est_achieved_occupancy = est_achieved_occupancy / max(sum_time, 1e-9)

    # 读取每个 Thread Block 最大允许的共享内存大小（字节）
    sharedMemPerBlock = data['deviceProperties'][0]['sharedMemPerBlock']
    # 读取每个 Thread Block 最大允许的寄存器数量
    regsPerBlock = data['deviceProperties'][0]['regsPerBlock']
    # 读取每个 Thread Block 最大允许的线程数
    maxThreadsPerBlock = data['deviceProperties'][0]['maxThreadsPerBlock']


    return node2kernels, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock
    #node2kernels是cpu算子node到gpu kernel实际执行的映射表


    #######绕了一大圈最后只用到了sharedMemPerBlock、regs、maxThread


def get_topo(fx_nodes, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock):
    nodes = {node.name: node for node in fx_nodes}
    in_degree = {node.name: 0 for node in nodes.values()}
    for node in nodes.values():
        for input_node in node.all_input_nodes:
            in_degree[node.name] += 1
    visited = set()
    result = []
    # print("fx_nodes", nodes.keys(), file=output_file)
    result = launch(nodes, result, in_degree, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock)

    #result中存放最终计算出的“最佳launch顺序”

    return result, nodes

def recompile(model_class_name, graph_module, inputs, apply_opara_schedule=True):
    
    path = os.path.abspath(os.path.dirname(__file__))
    # model_class_name = graph_module.__class__.__name__
    for i in inputs:
        model_class_name += "_" + str(i.shape)
    path += "/profile_result/" + model_class_name + ".pt.trace.json"
    if os.path.exists(path) is False:
        ModelProfiler.profile(graph_module, inputs, path)

    #获取ModelProfiler得到的json数据，剖析算子执行时间等信息
    node2kernels, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock = get_resource_from_json(path)

    for i, node in enumerate(graph_module.graph.nodes):
        if not hasattr(node, 'info'):
            if i < len(node2kernels):
                setattr(node, 'info', node2kernels[i])
            else:
                setattr(node, 'info', [])
    
    #node_profile 信息供 TCAS （优先调度关键路径算法）使用
    node_profiles = {}
    for i, node in enumerate(graph_module.graph.nodes):
        # 此时 graph.nodes 的第 i 个节点，严格对应 node2kernels 的第 i 个数据
        node_profiles[node.name] = node2kernels[i] if i < len(node2kernels) else []

    device_props = {
        'sharedMemPerBlock': sharedMemPerBlock,
        'regsPerBlock': regsPerBlock,
        'maxThreadsPerBlock': maxThreadsPerBlock
    }

    if apply_opara_schedule:
        #获取launch的顺序，存入result
        result, torch_nodes = get_topo(graph_module.graph.nodes, sharedMemPerBlock, regsPerBlock, maxThreadsPerBlock)


        #重构计算图，根据计算出的 result 顺序，物理修改 FX Graph 的链表连接
        #在 result[i] 这个节点后面，紧接着插入 result[i+1]
        #效果：这将强制改变代码的执行顺序，使其符合算出的“交错发射”策略
        size = len(result)
        for i in range(size - 1):
            torch_nodes[result[i]].append(torch_nodes[result[i+1]])
        # print(graph_module.graph, file=output_file)

        graph_module.graph.lint()
        graph_module.recompile()
    
    return node_profiles, device_props

#Opara使用的是计算密集与内存密集交替的launch
#我想使用关键路径优先的launch，然后结合stream分配，这里怎么结合是一个问题
#目前recompile是可以产生一个交替执行的launch，同时返回TCAS一个整体的算子执行信息，TCAS可以算出关键路径
