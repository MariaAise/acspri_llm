---
title: "Gradio Guide"
css: styles.css
author: "Maria A"
description: "Introduction to Gradio and its features."
tags: ["deep learning", "gradio", "research"]
---

# Gradio Guide

**GradIO** is a Python library that makes it easy to create web-based interfaces for machine learning/AI models and other Python functions. 

It is easy to use and highly customizable, making it a popular choice for researchers and developers who want to quickly prototype and share their work.

## Installation

```python
!pip install gradio
```

## Basic Usage

```python
import gradio as gr

## Defines a function `greet` that takes a name and an intensity value, and returns a greeting with that many exclamation marks.

def greet(name, intensity):
    return "Hello " + name + "!" * int(intensity)

demo=gr.Interface(
    fn=greet,
    inputs=["text", gr.Slider(1, 10)],
    outputs="text"  
)
```
`gr.Interface` is the main class in Gradio that allows you to create a web interface for your function.

In Gradio, the **inputs** and **outputs** parameters of `gr.Interface` define how data flows between the user interface and your function (`fn`):

**Inputs**:
Each item in the inputs list corresponds to an argument in your function:
- "**text**" creates a textbox for the name argument.
- **gr.Slider(1, 10)** creates a slider for the intensity argument. 
When the user enters a name and moves the slider, those values are passed to greet(name, intensity).


**Outputs**:
The value returned by your function is sent to the output component(s) defined in outputs.

- "text" means the result will be displayed as text.
If your function returns multiple values, use a list of output components.

```python
# Launch the interface
demo.launch(share=True, debug=True)
```

`demo.launch()` starts the Gradio web interface.

- It opens a local web server and displays your app in the browser.
Users can interact with your function(s) using the UI you defined.
You can add options like `share=True` to get a public link, or specify `server_name` and `server_port` for custom hosting.

## Building your app as Lego with `Blocks API`

To have more control over the interface layout and components, you can use the **Blocks API**. 
Compared to `gr.Interface`, which is more of a high-level abstraction, the Blocks API allows you to build complex and custom interfaces by using individual components and their interactions.

Here's an example of building a simple image classification app:

```python
import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your model (example with a simple function)
def predict_image(image):
    # This mocks-up the actual prediction to keep the example simple
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    # Simulate prediction
    return {"Cat": 0.7, "Dog": 0.3}

with gr.Blocks(title="Animal Classifier") as demo:
    gr.Markdown("# 🐾 Animal Classifier")
    gr.Markdown("Upload an image to classify as cat or dog.")

    image_input = gr.Image(type="pil", label="Upload Image")
    predict_btn = gr.Button("Classify")
    output_label = gr.Label()

    predict_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=output_label
    )

demo.launch(debug=True)
```
In this example, we use the following components:
- `gr.Blocks()`: The main container for the app.
- `gr.Markdown()`: To add titles and descriptions.
- `gr.Image()`: For image upload.
- `gr.Button()`: To trigger the prediction.
- `gr.Label()`: To display the prediction results.

`predict_btn.click(...)` sets up the interaction: when the button is clicked, it calls `predict_image` with the uploaded image and displays the result in the label.

------------

## Key Components

### Input Types
```python
# Text input
gr.Textbox(lines=2, placeholder="Enter text here")

# Number input
gr.Number(value=0, label="Age")

# Slider
gr.Slider(minimum=0, maximum=100, step=1, value=50, label="Intensity")

# Dropdown
gr.Dropdown(choices=["Option 1", "Option 2", "Option 3"], label="Choose")

# Checkbox
gr.Checkbox(label="Agree to terms")

# Radio buttons
gr.Radio(choices=["Yes", "No", "Maybe"], label="Answer")

# Image input
gr.Image(type="pil", label="Upload Image")

# File input
gr.File(label="Upload File")

# Audio input
gr.Audio(type="filepath", label="Record Audio")
```

### Output Types
```python
# Text output
gr.Textbox(label="Result")

# Label output (for classifications)
gr.Label(num_top_classes=3)

# Image output
gr.Image(label="Processed Image")

# JSON output
gr.JSON(label="Structured Data")

# Plot output
gr.Plot(label="Chart")
```

## Advanced Features

Now, let's have a look at how we use more gradio features to build more complex apps using HF models.


```python
import gradio as gr

def process_data(name, age, subscribe):
    result = f"Name: {name}, Age: {age}"
    subscription = "Subscribed" if subscribe else "Not subscribed"
    return result, subscription

with gr.Blocks(title="User Info Form") as demo:
    gr.Markdown("# User Information")
    with gr.Column():
        name = gr.Textbox(label="Name")
        age = gr.Number(label="Age")
        subscribe = gr.Checkbox(label="Subscribe to newsletter")
        submit_btn = gr.Button("Submit")
    with gr.Column():
        summary = gr.Textbox(label="Summary")
        sub_status = gr.Textbox(label="Subscription Status")

    submit_btn.click(
        fn=process_data,
        inputs=[name, age, subscribe],
        outputs=[summary, sub_status]
    )

demo.launch()
```

### 2. Chat Interface

```python
import random

def chatbot(message, history):
    responses = [
        "That's interesting!",
        "Tell me more about that.",
        "I understand how you feel.",
        "What else would you like to know?"
    ]
    return random.choice(responses)

demo = gr.ChatInterface(
    fn=chatbot,
    title="AI Chatbot",
    description="Chat with an AI assistant"
)
```

### 3. Blocks API (More Flexible)

```python
with gr.Blocks(title="Advanced Form") as demo:
    gr.Markdown("# User Registration Form")
    
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Full Name")
            email = gr.Textbox(label="Email Address")
        with gr.Column():
            age = gr.Slider(0, 100, label="Age")
            country = gr.Dropdown(["USA", "UK", "Canada", "Other"], label="Country")
    
    subscribe = gr.Checkbox(label="Subscribe to newsletter")
    submit = gr.Button("Register")
    
    output = gr.Textbox(label="Registration Summary")
    
    def register(name, email, age, country, subscribe):
        sub_status = "subscribed" if subscribe else "not subscribed"
        return f"Registered: {name} ({email}), {age} years, from {country}, {sub_status}"
    
    submit.click(
        fn=register,
        inputs=[name, email, age, country, subscribe],
        outputs=output
    )
```

### 4. Real-time Updates

```python
def update_slider_value(value):
    return f"Selected value: {value}"

with gr.Blocks() as demo:
    slider = gr.Slider(0, 100, label="Adjust value")
    output = gr.Textbox()
    
    slider.change(fn=update_slider_value, inputs=slider, outputs=output)
```

## Deployment Options

### 1. Local Development
```python
demo.launch()  # Opens in default browser
demo.launch(share=True)  # Creates public link
demo.launch(server_name="0.0.0.0", server_port=7860)  # Specific port
```

### 2. Hugging Face Spaces
```python
# Save your script and push to Hugging Face repository
# Gradio automatically detects and hosts your app
```



## Best Practices


- **Caching** (for expensive computations):
```python
@gr.cache()
def expensive_computation(param):
    # Heavy computation here
    return result
```

- **Theming**:
```python
demo = gr.Interface(theme=gr.themes.Soft())
```

- **Authentication**:
```python
demo.launch(auth=("username", "password"))
```

## Example Complete App

```python
import gradio as gr
import numpy as np

def calculator(a, b, operation):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else "Error: Division by zero"

demo = gr.Interface(
    fn=calculator,
    inputs=[
        gr.Number(label="First number"),
        gr.Number(label="Second number"),
        gr.Radio(["add", "subtract", "multiply", "divide"], label="Operation")
    ],
    outputs=gr.Number(label="Result"),
    title="Simple Calculator",
    description="Perform basic arithmetic operations",
    examples=[
        [5, 3, "add"],
        [10, 2, "divide"],
        [7, 4, "multiply"]
    ]
)

if __name__ == "__main__":
    demo.launch()
```

GradIO makes it incredibly easy to create interactive web interfaces for  Python code, if it's machine learning models, data processing scripts, or any other function you want to share with others.