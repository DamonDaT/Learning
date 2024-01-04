import pdfplumber
import pandas as pd

"""
1. Class pdfplumber.PDF

Attrs:
    .metadata: 一个由 PDF 的 Info 尾部信息中的元数据键/值对组成的字典，通常包括 "CreationDate," "ModDate," "Producer," 等等。
    .pages: 包含每个已加载页面的 pdfplumber.Page 实例的列表。
    
Methods:
    .close(): 默认情况下，Page 对象会缓存其布局和对象信息，以避免重新处理。然而，在解析大型 PDF 时，这些缓存的属性可能需要大量内存。你可以使用此方法来清除缓存并释放内存。

2. Class pdfplumber.Page

Attrs:
    .page_number: 顺序页码，从第一页开始为 1，第二页为 2，以此类推。
    .width: 页面的宽度。
    .height: 页面的高度。
    .objects / .chars / .lines / .rects / .curves / .images: 这些属性都是列表，每个列表包含页面上嵌入的每个此类对象的一个字典。
    
Methods:
    .to_image(): 将页面转化为 PageImage 类对象。
    .extract_text(): 将页面的所有字符对象汇集成一个单一的字符串。
    .extract_table(): 返回从页面上最大的表格中提取的文本。
    
3. Class pdfplumber.display.PageImage

Attrs:
    .resolution: 所需每英寸像素数。默认值：72。类型：整数。
    .width: 所需图像宽度（以像素为单位）。默认值：未设置，由分辨率确定。类型：整数。
    .height: 所需图像高度（以像素为单位）。默认值：未设置，由分辨率确定。类型：整数。
    .antialias: 是否在创建图像时使用抗锯齿。将其设置为True可以创建具有较少锯齿的文本和图形，但文件大小会更大。默认值：False。类型：布尔值。

Methods:
    .crop(): 返回裁剪到边界框的页面版本，边界框应表示为4元组，值为 (x0, top, x1, bottom)。
"""

# 加载 pdf 文件
pdf = pdfplumber.open(r'D:\XXX\YYY.pdf')

# 页面
pdf_pages = pdf.pages

# 利用 pandas.DataFrame 处理表格数据
table = pdf_pages[0].extract_table()
df = pd.DataFrame(data=table[1:], columns=table[0])

# 官方推荐的裁剪图像的方式
image = pdf_pages[1].images[0]
bbox = (image["x0"], image["top"], image["x1"], image["bottom"])
cropped_image = pdf_pages[1].crop(bbox=bbox)
