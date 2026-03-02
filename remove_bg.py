from PIL import Image

def remove_background(img_path, out_path):
    img = Image.open(img_path).convert("RGBA")
    datas = img.getdata()
    
    new_data = []
    # threshold for white/near-white
    for item in datas:
        # If it's mostly white and very bright, make transparent
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
            
    img.putdata(new_data)
    
    # Auto crop the transparent borders
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        
    img.save(out_path, "PNG")

remove_background(r"C:\Users\Gwendal\.gemini\antigravity\brain\105b285e-283a-414b-aa0d-1a38a2b61774\quant_lab_logo_notext_1772467659016.png", r"assets\logo.png")
print("Background removed and saved to assets/logo.png")
