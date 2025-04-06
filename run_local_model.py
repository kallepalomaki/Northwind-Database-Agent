import subprocess
from langchain.llms import BaseLLM
from pydantic import BaseModel
from langchain.schema import LLMResult, Generation  # This is a placeholder for a wrapped result

class LocalModelLLM(BaseLLM, BaseModel):
    exec_path: str  # Define as a Pydantic field
    model_file: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Let Pydantic handle initialization

    def _generate(self, prompt: str, stop: list = None) -> str:
        # Verify the exec_path is a valid string
        if not isinstance(self.exec_path, str) or not self.exec_path:
            raise ValueError(f"Invalid exec_path: {self.exec_path}")
        if isinstance(prompt, list):
            prompt = " ".join(prompt)

            # Verify the model_file is a valid string
        if not isinstance(self.model_file, str) or not self.model_file:
            raise ValueError(f"Invalid model_file: {self.model_file}")

        # Construct the command with the correct parameters
        command = [
            self.exec_path,
            '-m', self.model_file,  # Model file
            '-p', prompt,           # The input prompt
            '--threads', '4',       # Number of threads
            '--n-gpu-layers', '0',   # Disabling GPU layers
            '--n-predict', '50',
            '-st'
        ]

        # Debug: Print out the command
        print("Running command:", command)

        # Running the subprocess with the constructed command
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            # Create a Generation object for the model output
            generation = Generation(text=result.stdout.strip())

            # Return the output wrapped in an LLMResult object (list of lists of Generation objects)
            return LLMResult(
                generations=[[generation]]  # Wrap the Generation object in a list of lists
            )

        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e.stderr}")
            # Return an error message in the same structure
            error_generation = Generation(text=f"Error: {e.stderr.strip()}")
            return LLMResult(
                generations=[[error_generation]]  # Wrap error message in a list of lists
            )

        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            # Return an unexpected error in the same structure
            unexpected_error_generation = Generation(text=f"Error: {str(e)}")
            return LLMResult(
                generations=[[unexpected_error_generation]]  # Wrap unexpected error message in a list of lists
            )

    @property
    def _llm_type(self) -> str:
        # Return the type of LLM, e.g., 'local' for local models
        return "local"

# Example usage
model_path = '../llama.cpp/build/bin/llama-cli'  # Path to llama-cli
#model_file = '../models/deepseek/deepseek-coder-6.7b-instruct.Q4_K_M.gguf'  # Path to the model file
model_file="../models/deepseek/DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf"
#model_file = "../models/deepseek/deepseek-coder-1.3b-instruct-q4_k_m.gguf"
# Example usage
local_model = LocalModelLLM(exec_path=model_path, model_file=model_file)

# Now you can use it with LangChain's LLMChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

template = "Write a Python function to print 'Hello, World!'. Only plain code, no explanation."
prompt = PromptTemplate(input_variables=[], template=template)

llm_chain = LLMChain(prompt=prompt, llm=local_model)

# Run it
output = llm_chain.run({})
print(output)
