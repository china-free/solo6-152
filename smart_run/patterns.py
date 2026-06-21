"""Built-in error pattern library.

Each entry is a small rule: a ``regex`` matched against the captured tail, a
human-readable ``title`` and ``explanation`` in plain language, plus a list of
actionable ``suggestions``. The list is tuned toward the kinds of failures
that bite ML fine-tuning and large-scale data-cleaning jobs -- CUDA OOM,
segfaults, OOM-killer, missing files/modules, disk full, distributed/NCCL
errors, and so on.

Rules are evaluated in order; the first match wins, so put the more specific
ones near the top.
"""

from __future__ import annotations

from typing import Any, Dict, List


PATTERNS: List[Dict[str, Any]] = [
    {
        "id": "cuda_oom",
        "regex": r"CUDA out of memory|OutOfMemoryError.*CUDA|cudaMalloc.*failed",
        "severity": "critical",
        "title": "GPU 显存不足（CUDA Out of Memory）",
        "explanation": (
            "程序在向显卡申请显存时失败了——模型或单次处理的 batch 放不进当前 GPU 的显存。"
            "通常是 batch_size 过大、序列过长，或显存被别的进程占用。"
        ),
        "suggestions": [
            "调小 batch_size，或开启梯度累积（gradient accumulation）达到等效大 batch。",
            "启用混合精度训练（fp16 / bf16）以降低显存占用。",
            "减小最大序列长度 / 序列截断长度。",
            "用 LoRA、QLoRA 等参数高效微调替代全量微调。",
            "用 nvidia-smi 检查是否有残留进程占着显存，必要时 kill 掉。",
            "启用 PyTorch 的显存碎片整理：PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。",
        ],
    },
    {
        "id": "host_oom_killer",
        "regex": r"\bKilled\b|(?<!CUDA )out of memory|Cannot allocate memory|MemoryError|cgroup.*killed",
        "severity": "critical",
        "title": "主机内存不足（被 OOM Killer 杀掉）",
        "explanation": (
            "进程占用的内存超过了系统可用内存，被操作系统的 OOM Killer 直接杀掉，"
            "通常表现为退出码 137 且没有 Python 异常堆栈。常见原因是 DataLoader 的 "
            "num_workers 过大、数据被整体载入内存、或内存泄漏。"
        ),
        "suggestions": [
            "降低 DataLoader 的 num_workers，或改用流式 / 分块读取数据。",
            "检查是否把整个数据集一次性 load 进了内存，改用迭代式加载。",
            "用 free -h / htop 观察内存峰值，必要时换更大内存的机器。",
            "如果是多进程，检查是否有子进程泄漏（僵尸进程堆积）。",
        ],
    },
    {
        "id": "segfault",
        "regex": r"Segmentation fault|SIGSEGV|core dumped|exit code 139|returned 139",
        "severity": "critical",
        "title": "段错误（Segmentation Fault）",
        "explanation": (
            "程序发生了非法内存访问，通常是 C/C++ 扩展、CUDA 内核或底层库崩溃，"
            "而不是 Python 层的异常，因此往往只有一串看不懂的 C++ 堆栈。"
        ),
        "suggestions": [
            "核对 PyTorch / CUDA / cuDNN / 驱动版本是否互相兼容。",
            "降低 num_workers（多进程 DataLoader 的共享内存问题是常见元凶）。",
            "打开 faulthandler：python -X faulthandler train.py，以获取 Python 层堆栈。",
            "如使用了自定义 C++/CUDA 算子，检查其内存管理与边界条件。",
            "尝试在 CPU 模式下复现，排除硬件 / 驱动问题。",
        ],
    },
    {
        "id": "file_not_found",
        "regex": r"FileNotFoundError|No such file or directory|\[Errno 2\]",
        "severity": "error",
        "title": "找不到文件（FileNotFoundError）",
        "explanation": "程序要读写的文件或目录不存在，多半是路径写错、相对路径基准不对，或数据未下载到位。",
        "suggestions": [
            "确认路径是相对脚本运行目录还是绝对路径，建议用绝对路径。",
            "检查数据集 / 权重文件是否已下载、是否在指定位置。",
            "若用相对路径，注意 cwd 是 smart-run 的启动目录而非脚本所在目录。",
        ],
    },
    {
        "id": "module_not_found",
        "regex": r"ModuleNotFoundError|ImportError|No module named",
        "severity": "error",
        "title": "缺少依赖（ModuleNotFoundError）",
        "explanation": "代码 import 的某个包在当前环境里没装，或装到了别的 Python 环境里。",
        "suggestions": [
            "按报错信息 pip/conda install 对应的包。",
            "确认用的是对的解释器 / 虚拟环境（which python, conda env list）。",
            "若是自己写的模块，检查 PYTHONPATH / 包结构 / __init__.py 是否就位。",
        ],
    },
    {
        "id": "disk_full",
        "regex": r"No space left on device|OSError.*\b28\b|disk full|ENOSPC",
        "severity": "critical",
        "title": "磁盘空间不足（No space left on device）",
        "explanation": "写文件（checkpoint、缓存、日志、tokenize 缓存）时磁盘满了。",
        "suggestions": [
            "用 df -h 检查磁盘占用，清理 checkpoints / 缓存目录。",
            "减少保存 checkpoint 的频率或只保留最近 N 个。",
            "把缓存目录（如 HuggingFace 的 ~/.cache）指到大磁盘并设环境变量。",
        ],
    },
    {
        "id": "nccl_error",
        "regex": r"NCCL error|ncclUnhandledCudaError|NCCL.*timeout|rank.*failed|backend.*gloo",
        "severity": "critical",
        "title": "分布式通信失败（NCCL / 多卡）",
        "explanation": "多卡或多节点训练时 GPU 间通信出错，常见原因是网络/IB 配置、某些 rank 挂了、或端口被占。",
        "suggestions": [
            "设置 NCCL_DEBUG=INFO 获取更详细日志定位具体 rank。",
            "检查所有 rank 是否都正常、是否有某张卡 OOM 掉了导致整体超时。",
            "确认主节点端口未被占用、防火墙放行了 NCCL 端口。",
            "多机时核对网络接口：NCCL_SOCKET_IFNAME=eth0 之类。",
        ],
    },
    {
        "id": "cuda_driver_mismatch",
        "regex": r"CUDA driver version is insufficient|CUDA runtime version|nvidia-smi.*failed|cuda version mismatch",
        "severity": "error",
        "title": "CUDA / 驱动版本不匹配",
        "explanation": "PyTorch 编译时链接的 CUDA 版本与机器上的显卡驱动不兼容。",
        "suggestions": [
            "用 nvidia-smi 看驱动支持的最高 CUDA 版本。",
            "安装与驱动匹配的 PyTorch 版本（参考 pytorch.org 的版本表）。",
            "若在容器内，确认基础镜像的 CUDA 版本与宿主机驱动兼容。",
        ],
    },
    {
        "id": "device_side_assert",
        "regex": r"device-side assert|CUDA error: an illegal memory access|illegal memory access",
        "severity": "critical",
        "title": "GPU 端断言失败（device-side assert）",
        "explanation": "GPU kernel 内部触发断言，最常见原因是索引越界，例如 label id 超出了 vocab/类别数。",
        "suggestions": [
            "检查 label 是否 >= num_labels / vocab_size，是否从 0 开始且无负数。",
            "设置 CUDA_LAUNCH_BLOCKING=1 让报错更精确地指到出错位置。",
            "核对 tokenizer 与模型的词表是否一致（padding/unk token id）。",
        ],
    },
    {
        "id": "keyerror",
        "regex": r"KeyError",
        "severity": "error",
        "title": "字典键不存在（KeyError）",
        "explanation": "代码访问了字典里不存在的键，常出现在配置 / 数据字段缺失时。",
        "suggestions": [
            "按报错补上缺失的键，或在取值前用 .get(key, default)。",
            "检查数据预处理是否漏了某个字段、配置文件是否缺少某项。",
        ],
    },
    {
        "id": "shape_mismatch",
        "regex": r"RuntimeError.*size mismatch|Expected.*but got|shape.*mismatch|invalid argument",
        "severity": "error",
        "title": "张量形状不匹配（Shape Mismatch）",
        "explanation": "矩阵/张量运算的维度对不上，通常是数据维度、模型层配置或 batch padding 出错。",
        "suggestions": [
            "按报错对照 expected vs got 的维度，定位是哪一层输入维度不对。",
            "检查 padding / collate 函数是否产生了不一致的形状。",
            "确认模型 config（hidden_size、num_heads 等）与权重一致。",
        ],
    },
    {
        "id": "value_error",
        "regex": r"ValueError",
        "severity": "error",
        "title": "取值非法（ValueError）",
        "explanation": "传入了不合法的值，例如 batch_size 为 0、空列表、维度为空等。",
        "suggestions": [
            "按报错检查相关参数取值范围与边界情况。",
            "确认数据非空、采样器返回了有效的 batch。",
        ],
    },
    {
        "id": "assertion_error",
        "regex": r"AssertionError|Assertion failed",
        "severity": "error",
        "title": "断言失败（AssertionError）",
        "explanation": "代码里某处 assert 条件不成立，通常用于防御性检查，说明前置条件没满足。",
        "suggestions": [
            "看断言所在文件/行号，理解它在校验什么前置条件。",
            "按断言信息修正输入数据或参数。",
        ],
    },
    {
        "id": "permission_denied",
        "regex": r"PermissionError|Permission denied|\[Errno 13\]",
        "severity": "error",
        "title": "权限不足（Permission denied）",
        "explanation": "程序要读写的文件或目录没有权限访问。",
        "suggestions": [
            "检查文件/目录的属主与权限（ls -l / chmod）。",
            "确认不是在没权限的共享路径下写 checkpoint / 日志。",
        ],
    },
    {
        "id": "network",
        "regex": r"ConnectionError|ConnectionRefused|Connection timed out|ConnectionResetError|URLError|SSLError|ProxyError|Temporary failure in name resolution",
        "severity": "error",
        "title": "网络连接异常",
        "explanation": "程序访问网络资源失败，常见于下载模型/数据、连 wandb、连对象存储时。",
        "suggestions": [
            "检查网络 / 代理设置（HTTP_PROXY, HTTPS_PROXY）。",
            "若需离线，提前把模型/数据下载到本地并设置 HF_HOME 等缓存变量。",
            "确认服务地址与端口、鉴权 token 是否正确。",
        ],
    },
    {
        "id": "timeout",
        "regex": r"TimeoutError|timed out|DeadlineExceeded",
        "severity": "error",
        "title": "操作超时（Timeout）",
        "explanation": "某个操作在限定时间内没完成，可能是数据加载、网络请求或分布式 barrier 超时。",
        "suggestions": [
            "适当调大对应超时阈值。",
            "检查数据加载是否过慢、网络是否抖动、是否有 rank 卡住。",
        ],
    },
    {
        "id": "syntax_error",
        "regex": r"SyntaxError|IndentationError|TabError",
        "severity": "error",
        "title": "代码语法错误（SyntaxError）",
        "explanation": "脚本本身有语法或缩进错误，根本没跑起来。",
        "suggestions": [
            "按报错的文件和行号修正语法 / 缩进。",
            "若最近改过代码，先 review 一下 diff。",
        ],
    },
    {
        "id": "keyboard_interrupt",
        "regex": r"KeyboardInterrupt",
        "severity": "warning",
        "title": "被手动中断（KeyboardInterrupt）",
        "explanation": "进程被 Ctrl-C 手动终止，不是真正的崩溃，但你可能想知道它停了。",
        "suggestions": [
            "如果是误触，重新启动即可。",
            "如果训练已保存 checkpoint，可从最近 checkpoint 续训。",
        ],
    },
    {
        "id": "generic_python_traceback",
        "regex": r"Traceback \(most recent call last\)",
        "severity": "error",
        "title": "Python 抛出了未捕获异常",
        "explanation": "程序抛出了异常且没有被 try/except 捕获。请查看 tail 最后一个 `XXXError: <message>` 行确定具体异常类型与原因。",
        "suggestions": [
            "定位 tail 中最后一个异常类型与消息行，并向上找到首次抛出的位置。",
            "把该行 + 调用栈关键几行贴出来进一步排查。",
        ],
    },
]


def match_all(tail: str) -> List[Dict[str, Any]]:
    """Return every rule that matches (ordered). Useful for multi-cause cases."""
    import re

    hits: List[Dict[str, Any]] = []
    for rule in PATTERNS:
        try:
            if re.search(rule["regex"], tail, re.IGNORECASE | re.MULTILINE):
                hits.append(rule)
        except re.error:
            continue
    return hits
