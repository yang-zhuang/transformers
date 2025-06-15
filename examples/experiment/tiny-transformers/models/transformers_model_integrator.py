import os
import re
import glob
import json
import ast
from huggingface_hub import hf_hub_download


def integrate_model(repo_id, files_to_download, dir_name, transformers_path="../../../../"):
    """
    通用模型集成函数

    参数:
    repo_id (str): Hugging Face 仓库 ID (如 "Qwen/Qwen-7B-Chat")
    files_to_download (list): 需要下载的文件列表 (如 ["configuration.py", "modeling.py"])
    dir_name (str): 模型目录名称 (如 "qwen")
    transformers_path (str): Transformers 库根路径 (默认为 "../../../../")
    """
    # =============== 步骤 1: 准备模型目录 ===============
    print("\n" + "=" * 50)
    print(f"步骤 1: 创建模型目录 '{dir_name}'")
    print("=" * 50)

    models_dir = os.path.join(transformers_path, "src/transformers/models/")
    target_dir = os.path.join(models_dir, dir_name)

    # 创建模型目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    print(f"✅ 模型目录: {target_dir}")

    # =============== 步骤 2: 下载模型文件 ===============
    print("\n" + "=" * 50)
    print(f"步骤 2: 下载模型文件 ({len(files_to_download)} 个文件)")
    print("=" * 50)

    status = download_model_files(repo_id, target_dir, files_to_download)

    if not status:
        import sys
        sys.exit()

    # =============== 步骤 3: 创建模型 __init__.py ===============
    print("\n" + "=" * 50)
    print(f"步骤 3: 创建模型 __init__.py")
    print("=" * 50)

    model_classes = create_model_init_file(target_dir)
    print(f"✅ 导出的类: {', '.join(model_classes)}")

    # =============== 步骤 4: 更新 models/__init__.py ===============
    print("\n" + "=" * 50)
    print(f"步骤 4: 更新 models/__init__.py")
    print("=" * 50)

    models_init = os.path.join(models_dir, "__init__.py")
    update_models_init_file(dir_name, models_init)

    # =============== 步骤 5: 更新 transformers/__init__.py ===============
    print("\n" + "=" * 50)
    print(f"步骤 5: 更新 transformers/__init__.py")
    print("=" * 50)

    top_level_init = os.path.join(transformers_path, "src/transformers/__init__.py")
    update_top_level_init_file(dir_name, model_classes, top_level_init)

    print("\n" + "=" * 50)
    print(f"✅ 所有步骤完成! '{dir_name}' 模型已成功集成到 Transformers 库")
    print("=" * 50)


def download_model_files(repo_id, target_dir, files_to_download):
    """从 Hugging Face Hub 下载模型文件"""
    print(f"从仓库 {repo_id} 下载文件到 {target_dir}")

    for filename in files_to_download:
        print(f"📥 下载: {filename}")
        try:
            # 下载文件到指定目录
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=target_dir,
                repo_type="model"
            )
            print(f"✅ 下载成功: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ 下载失败 {filename}: {str(e)}")
            # 尝试备用下载方式
            print("尝试备用下载方式...")
            try:
                backup_download(repo_id, filename, target_dir)
            except Exception as e2:
                print(f"❌ 备用下载也失败: {str(e2)}")

                return False

    return True


def backup_download(repo_id, filename, target_dir):
    """备用下载方式：直接通过 URL 下载"""
    import requests
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}?download=true"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    file_path = os.path.join(target_dir, filename)
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"✅ 备用下载成功: {filename}")


def create_model_init_file(target_dir):
    """创建模型目录的 __init__.py 文件"""
    print(f"创建 __init__.py 文件")

    # 收集所有模型文件（除了 __init__.py）
    model_files = glob.glob(os.path.join(target_dir, "*.py"))
    model_files = [f for f in model_files if not f.endswith("__init__.py")]

    # 提取所有公共类
    all_classes = []
    class_imports = {}

    for file_path in model_files:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        classes = extract_classes_from_file(file_path)

        if classes:
            class_imports[module_name] = classes
            all_classes.extend(classes)

    # 如果没有提取到类，使用默认值
    if not all_classes:
        print("⚠️ 未提取到任何类，使用回退值")
        # 尝试从文件名推断类名
        default_classes = {}
        for file_path in model_files:
            name = os.path.splitext(os.path.basename(file_path))[0]
            prefix = name.split('_')[0].capitalize()
            if "config" in name:
                default_classes[name] = [f"{prefix}Config"]
            elif "model" in name:
                default_classes[name] = [
                    f"{prefix}ForCausalLM",
                    f"{prefix}Model",
                    f"{prefix}PreTrainedModel"
                ]
            elif "token" in name:
                default_classes[name] = [f"{prefix}Tokenizer"]

        if not default_classes:
            default_classes = {"modeling": ["NewModel"]}

        for module, classes in default_classes.items():
            class_imports[module] = classes
            all_classes.extend(classes)

    # 生成导入语句
    import_lines = [
        f"from .{module} import {', '.join(classes)}"
        for module, classes in class_imports.items()
    ]

    # 创建 __init__.py 内容
    import_lines = "\n".join(import_lines)
    init_content = f"""# 此文件由脚本自动生成
# 请勿手动修改，如需更新请重新运行生成脚本

{import_lines}

__all__ = [
    {', '.join([f'"{c}"' for c in all_classes])}
]
"""

    # 写入文件
    init_file = os.path.join(target_dir, "__init__.py")
    with open(init_file, "w", encoding="utf-8") as f:
        f.write(init_content)

    print(f"✅ 已创建: {init_file}")

    return all_classes


def extract_classes_from_file(file_path):
    """从 Python 文件中提取所有公共类定义"""
    classes = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 使用正则表达式匹配类定义
        pattern = r"^class\s+(\w+)\s*(?:\(|:)"
        matches = re.findall(pattern, content, re.MULTILINE)

        # 过滤掉私有类（以_开头的）
        classes = [m for m in matches if not m.startswith('_')]
    except Exception as e:
        print(f"⚠️ 解析文件失败 {file_path}: {str(e)}")

    return classes


def update_models_init_file(model_name, models_init_path):
    """更新 models/__init__.py 文件"""
    if not os.path.exists(models_init_path):
        print(f"⚠️ 文件不存在: {models_init_path}")
        return

    print(f"更新 {models_init_path} 以包含 {model_name} 模型")

    # 读取文件内容
    with open(models_init_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查是否已存在导入语句
    import_statement = f"from .{model_name} import *\n"
    if import_statement in content:
        print(f"✅ {model_name} 已存在于 {models_init_path}")
        return

    # 在文件末尾添加导入语句
    with open(models_init_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(import_statement)

    print(f"✅ 已添加 '{import_statement.strip()}'")


def update_top_level_init_file(model_name, model_classes, top_level_path):
    """更新顶层 __init__.py 文件"""
    if not os.path.exists(top_level_path):
        print(f"⚠️ 文件不存在: {top_level_path}")
        return

    print(f"更新 {top_level_path} 以包含 {model_name} 模型")

    # 读取文件内容
    with open(top_level_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查是否已存在导入语句
    if f"from .models.{model_name} import *\n" in content:
        print(f"✅ {model_name} 已存在于 {top_level_path}")
        return

    # 准备要追加的内容
    key = f"models.{model_name}"
    append_content = f"""
# ===== 自动添加的模型: {model_name} =====
try:
    # 尝试添加到现有结构
    _import_structure["{key}"].extend(
{json.dumps(model_classes, ensure_ascii=False, indent=8)}
    )
except KeyError:
    # 创建新结构
    _import_structure["{key}"] = {json.dumps(model_classes, ensure_ascii=False, indent=4)}

# 确保在类型检查时导入
if TYPE_CHECKING:
    from .models.{model_name} import *
\n
"""
    append_content += """else:
    import sys

    _import_structure = {k: set(v) for k, v in _import_structure.items()}

    import_structure = define_import_structure(Path(__file__).parent / "models", prefix="models")
    import_structure[frozenset({})].update(_import_structure)

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )"""
    # 在文件末尾添加内容
    with open(top_level_path, "a", encoding="utf-8") as f:
        f.write(append_content)

    print(f"✅ 成功添加 {model_name} 模型到 {top_level_path}")


if __name__ == "__main__":
    # ===================== 用户配置区域 =====================
    # 只需修改这里的三个参数即可
    REPO_ID = "THUDM/chatglm3-6b"  # Hugging Face 模型仓库 ID
    FILES_TO_DOWNLOAD = [  # 需要下载的文件列表
        "configuration_chatglm.py",
        "modeling_chatglm.py",
        "tokenization_chatglm.py",
    ]
    MODEL_DIR_NAME = "chatglm3"  # 模型目录名称（小写）

    # ===================== 执行集成 =====================
    integrate_model(
        repo_id=REPO_ID,
        files_to_download=FILES_TO_DOWNLOAD,
        dir_name=MODEL_DIR_NAME
    )