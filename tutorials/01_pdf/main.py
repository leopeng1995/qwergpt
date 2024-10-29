import logging
from pathlib import Path
from tqdm import tqdm

from qwergpt.utils import convert_single_pdf


def main():
    pdf_dir = Path("./input")
    output_root = Path("./output")
    
    # 确保输出目录存在
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 获取所有PDF文件并检查
    pdf_files = list(pdf_dir.glob("*.PDF"))
    if not pdf_files:
        logging.error(f"在目录 {pdf_dir} 中没有找到PDF文件")
        return

    logging.info(f"找到 {len(pdf_files)} 个PDF文件")
    logging.info(f"PDF文件列表: {[f.name for f in pdf_files]}")
    
    # 检查目录是否存在
    if not pdf_dir.exists():
        logging.error(f"PDF目录不存在: {pdf_dir}")
        return

    # 顺序处理每个PDF文件
    for pdf_file in tqdm(pdf_files, desc="正在处理PDF文件"):
        convert_single_pdf(pdf_file, output_root)

    logging.info("所有PDF文件转换完成")


if __name__ == "__main__":
    main()
