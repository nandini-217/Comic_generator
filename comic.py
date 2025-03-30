import os
import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Can be replaced with LLaMA or other local model
IMAGE_MODEL = "stabilityai/stable-diffusion-2-1"
COMIC_FONT = "arial.ttf"  # Replace with comic-style font if available
OUTPUT_DIR = "comic_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize models (with caching)
@st.cache_resource
def load_models():
    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load Stable Diffusion
    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        IMAGE_MODEL,
        torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    return llm_pipeline, sd_pipeline

def generate_story(prompt, llm_pipeline):
    """Generate a structured story with LLM"""
    system_prompt = """You are a comic book writer. Generate a short comic story with:
1. Introduction: Set up the scene and characters
2. Storyline: Develop the plot
3. Climax: The most exciting part
4. Moral: The lesson or conclusion

Format your response clearly with these headings."""
    
    full_prompt = f"{system_prompt}\n\nUser Prompt: {prompt}\n\nStory:"
    
    response = llm_pipeline(
        full_prompt,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    return response[0]['generated_text']

def parse_story(story_text):
    """Parse the generated story into sections"""
    sections = {
        "introduction": "",
        "storyline": "",
        "climax": "",
        "moral": ""
    }
    
    current_section = None
    for line in story_text.split('\n'):
        line_lower = line.lower()
        if "introduction" in line_lower:
            current_section = "introduction"
        elif "storyline" in line_lower:
            current_section = "storyline"
        elif "climax" in line_lower:
            current_section = "climax"
        elif "moral" in line_lower:
            current_section = "moral"
        elif current_section and line.strip():
            sections[current_section] += line + '\n'
    
    return sections

def generate_image(prompt, sd_pipeline, section_name):
    """Generate comic-style image for a story section"""
    comic_prompt = f"Comic book style, vibrant colors, {prompt}"
    image = sd_pipeline(
        comic_prompt,
        negative_prompt="blurry, low quality, sketch",
        num_inference_steps=30
    ).images[0]
    
    # Save image
    image_path = os.path.join(OUTPUT_DIR, f"{section_name}.png")
    image.save(image_path)
    return image, image_path

def create_comic_page(text, image_path, page_num):
    """Combine text and image into a comic page"""
    # Load image
    image = Image.open(image_path)
    width, height = image.size
    
    # Create a larger canvas for text
    new_height = height + 200  # Space for text
    comic_page = Image.new('RGB', (width, new_height), 'white')
    comic_page.paste(image, (0, 0))
    
    # Add text
    try:
        font = ImageFont.truetype(COMIC_FONT, 16)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(comic_page)
    text_position = (20, height + 10)
    draw.text(text_position, text, fill="black", font=font)
    
    # Save comic page
    output_path = os.path.join(OUTPUT_DIR, f"comic_page_{page_num}.png")
    comic_page.save(output_path)
    return comic_page, output_path

def main():
    st.title("ComicCrafter AI")
    st.write("Generate your own comic book story!")
    
    user_prompt = st.text_area("Enter your story prompt:", "A superhero dog saves the city from alien invasion")
    
    if st.button("Generate Comic"):
        with st.spinner("Loading AI models..."):
            llm_pipeline, sd_pipeline = load_models()
        
        with st.spinner("Generating your story..."):
            story_text = generate_story(user_prompt, llm_pipeline)
            st.subheader("Generated Story")
            st.write(story_text)
            
            sections = parse_story(story_text)
        
        st.subheader("Generating Comic Pages")
        cols = st.columns(2)
        
        for i, (section_name, section_text) in enumerate(sections.items(), 1):
            with st.spinner(f"Creating {section_name} image..."):
                image, image_path = generate_image(
                    f"{user_prompt}, {section_text[:100]}",  # Truncate text for image prompt
                    sd_pipeline,
                    section_name
                )
                
                comic_page, page_path = create_comic_page(
                    section_text,
                    image_path,
                    i
                )
                
                # Display in Streamlit
                cols[(i-1)%2].image(comic_page, caption=f"Page {i}: {section_name.capitalize()}", use_column_width=True)
        
        st.success("Your comic book is ready!")

if __name__ == "__main__":
    main()


# Alternative with smaller models
SMALL_LLM = "HuggingFaceH4/zephyr-7b-alpha"
SMALL_IMAGE_MODEL = "CompVis/stable-diffusion-v1-4"

# Replace the load_models function with:
@st.cache_resource
def load_models_light():
    # Smaller LLM
    llm_pipeline = pipeline(
        "text-generation",
        model=SMALL_LLM,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Smaller SD model
    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        SMALL_IMAGE_MODEL,
        torch_dtype=torch.float16,
        revision="fp16"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    return llm_pipeline, sd_pipeline