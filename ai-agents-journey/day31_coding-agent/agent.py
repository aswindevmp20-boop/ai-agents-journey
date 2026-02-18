from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_code(filepath):
    with open(filepath,"r") as f:
        return f.read()
    
def analyze_code(code):
    prompt = f"""
You are a senior python code reviewer.

Analyze the following code:
- Identify issues
- Suggest improvements
- Provide refactored version
- Explain why changes improve quality

code:
{code}
"""
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role":"user", "content": prompt}]
    )
    return response.choices[0].message.content

def run_agent(filepath):
    code = load_code(filepath)

    print("\n--- ORIGINAL CODE ---")
    print(code)

    print("\n-- ANALYSIS & REFRACTOR --")
    result = analyze_code(code)
    print(result)

run_agent("sample_code.py")


# Expected Output:

# --- ORIGINAL CODE ---
# def calculate_total(price, tax):
#     total = 0
#     total = price + tax
#     return total

# def print_user(name):
#     print("User name is " + name)


# -- ANALYSIS & REFRACTOR --
# **Code Analysis**

# The provided code consists of two simple functions: `calculate_total` and `print_user`. While the code works as intended, there are several issues and areas for improvement.

# ### Issues:

# 1. **Unnecessary variable initialization**: In the `calculate_total` function, the `total` variable is initialized to 0, only to be immediately reassigned to the sum of `price` and `tax`. This initialization is unnecessary.
# 2. **Inconsistent naming conventions**: The function names `calculate_total` and `print_user` follow the conventional underscore notation, but the variable names `price` and `tax` do not. Although they are camelCase, Python's official style guide (PEP 8) recommends using lowercase with words separated by underscores.
# 3. **Lack of input validation**: The `calculate_total` function does not check if the inputs `price` and `tax` are valid numbers. This could lead to errors if non-numeric values are passed.
# 4. **String concatenation**: In the `print_user` function, string concatenation is used to construct the output string. While this works, it is not the most efficient or readable way to build strings in Python.      
# 5. **No documentation**: The functions lack docstrings, making it difficult for other developers to understand their purpose, parameters, and return values.

# ### Suggested Improvements:

# 1. **Simplify the `calculate_total` function**: Remove the unnecessary variable initialization and consider adding input validation.
# 2. **Use consistent naming conventions**: Rename variables to follow the conventional underscore notation.
# 3. **Use f-strings for string formatting**: Replace string concatenation with f-strings for more readable and efficient string building.        
# 4. **Add docstrings**: Include docstrings in each function to provide documentation and improve code readability.
# 5. **Consider type hints**: Add type hints for function parameters and return types to improve code clarity and enable static type checking.    

# ### Refactored Code

# ```python
# def calculate_total(price: float, tax: float) -> float:
#     """
#     Calculate the total price by adding the tax to the base price.      

#     Args:
#         price (float): The base price.
#         tax (float): The tax amount.

#     Returns:
#         float: The total price.
#     """
#     if not isinstance(price, (int, float)) or not isinstance(tax, (int, float)):
#         raise TypeError("Both price and tax must be numbers")
#     return price + tax

# def print_user(name: str) -> None:
#     """
#     Print a greeting message with the user's name.

#     Args:
#         name (str): The user's name.
#     """
#     print(f"User name is {name}")
# ```

# ### Explanation of Changes

#  with f-strings in the `print_user` function for more readable and efficient string building.
# 4. **Docstrings added**: Docstrings are included in each function to provide documentation and improve code readability.
# 5. **Type hints added**: Type hints are added for function parameters and return types to improve code clarity and enable static type checking.

# These changes improve the overall quality of the code by making it more readable, maintainable, and efficient. The input validation and docstrings provide a clear understanding of the functions' behavior, and the f-strings and type hints make the code more concise and self-documenting.