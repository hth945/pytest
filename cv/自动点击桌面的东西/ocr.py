#%%
from PIL import Image
import pytesseract
 
image = Image.open('yy.jpg')
content = pytesseract.image_to_string(image, lang='chi_sim')   # 解析图片
# content = pytesseract.image_to_string(image)   # 解析图片
print(content)

# %%
