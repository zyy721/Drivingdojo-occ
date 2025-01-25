# import os

# # 调用 bash 脚本
# bash_script = "examples/world_model/scripts/run_video_svd_dojo_fsdp.sh"
# return_code = os.system(f"bash {bash_script}")

# if return_code != 0:
#     print("脚本运行出错，返回代码:", return_code)


import subprocess

# 指定要使用的 Conda 环境名称
conda_env = "dojo"  # 替换为您的环境名称
bash_script = "examples/world_model/scripts/run_video_svd_nusc_fsdp.sh"  # 替换为您的脚本路径

# 使用 conda run 激活环境并运行 Bash 脚本
result = subprocess.run(
    ["conda", "run", "-n", conda_env, "bash", bash_script], 
    capture_output=True, 
    text=True
)

# 打印输出和错误信息
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)

# 检查返回代码
if result.returncode != 0:
    print("脚本运行出错，返回代码:", result.returncode)
