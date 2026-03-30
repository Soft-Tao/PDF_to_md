# PDF_to_md
PDF转换成md，处理过电动力学（A）讲义

## 工作流
1. 调用本地运行的MinerU，借助其精准的图文分块能力，得到最终结果文档的基础架构。
```
mineru -p <input_path> -o <output_path> -b pipeline # 基于CPU运行
```
> MinerU repo: *https://github.com/opendatalab/mineru*，建议使用`uv pip`安装在虚拟环境里
2. 运行`convert_pdf.py`，该程序调用具备图形理解能力的大模型（如`qwen3-vl-plus`），对整个原始PDF文档进行识别，得到文字和公式识别更准确的参考文档。
```
python convert_pdf.py <input_path>
```
> 阿里云：*https://www.aliyun.com*
3. 运行`merge_refine.py`，该程序首先调用具备图形理解能力的LLM，对MinerU分区得到的代码块进行独立识别，结合第二部给出的参考文档，给出准确率最好的数学公式代码块；然后调用文本处理LLM（如`qwen3-max`，根据用户prompt要求对最终文档进行精修。
```
python merge_refine.py --mineru-dir <mineru's output dir> --reference <path to step2's output>
```
