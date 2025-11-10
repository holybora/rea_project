from pydantic import BaseModel, Field

# Define the SubmitFinalAnswer class
class SubmitFinalAnswer(BaseModel):
    """
    A Pydantic model representing the final answer to be submitted to the user.
    """
    final_answer: str = Field(..., description="The final answer to the user")