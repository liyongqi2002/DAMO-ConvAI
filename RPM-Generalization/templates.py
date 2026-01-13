
system_prompt_for_SThought = """Please generate a thought process that outlines what characteristics a "golden response" should have, based on the given input information, following the provided format template."""
base_prompt_for_SThought = """
# Task Guidelines
Below, I will provide you with the input information for this task, including the user persona , agent character , and dialogue context.
Please generate a thought process based on the provided Format Template of Generated Thought , using the input information I give you.

# Task Completion
## Input Information
### User Persona
'''
{user_persona}
'''

### Agent Character
'''
{agent_character}
'''

### Dialogue Context
'''
{str_dialogue_context}
'''

## Reference Golden Agent Response
'''
{agent_golden_response}
'''

> System: Please output the thought process in the specified format below. 

## Format Template of Generated Thought

### Part 1: Restatement of Key Information

#### Key Information in User Persona  
- **Interests and Values**: 
- **Aesthetic Preferences and Lifestyle**: 
- **Daily Habits and Behavioral Traits**: 
- **Professional Background and Role Values**: 
- **Recent Experience and Triggering Event**: 
- **Current Needs and Personal Alignment**: 

#### Key Information in Agent Character
- **Basic Identity and Background**: 
- **Occupation and Expertise**:  
- **Personality and Psychological Profile**: 
- **Language Style and Communication Approach**: 
- **Interests and Lifestyle Preferences**: 
- **Relationships and Social Behavior**: 
- **Values and Life Philosophy**: 

#### Key Information in Dialogue Context  
- **User's Past Engagement and State**:
- **Agent's Role and Response Style**: 
- **Current Dialogue Direction**: 

### Summary of Key Information

### Part 2: Iterative Revision

#### Trial 1  
**Initial attempt at capturing the essence of a strong response:**  
... (the detailed thinking process concerning the content the expected response is omitted, e.g., the expected golden response should include ... ) ...

**Based on this, the response might look like:**  
> ""

Verification: 
Revision Suggestion: 

#### Trial 2  
**Refined understanding of what makes a great response:**  
... (the detailed thinking process concerning the style the expected response is omitted, e.g., the expected golden response should be ... ) ...

**Now the response could be:**  
> ""

Verification: 
Revision Suggestion: 

#### Trial 3  
**Final integration of all essential elements for a perfect agent response:**  
... (the detailed thinking process is omitted) ...

**The response may become:**  
> ""

Verification: 
Revision Suggestion: 

### Part 3: Final Feature Set of the Golden Response  

Here are all the essential features that should be included in the final golden response:

```
[Core Features of the Golden Response]

## I. Content Characteristics

### 1. Alignment with User Persona  
- 
- 

### 2. Embodiment of Agent Character  
- 
- 

### 3. Continuity within Dialogue Context  
- 
- 


## II. Style Characteristics

### 1. Tone and Language Suitable for the User  
- 
- 

### 2. Expression Consistent with Agent Character  
- 
- 

### 3. Naturalness within Dialogue Flow  
- 
- 
```

## Generated Thought
"""



system_prompt_for_model_pxy ="""Please complete the agent response part after thinking the core features of the golden response to respond."""

base_prompt_for_model_pxy ="""
# Task Completion
## Input Information
### User Persona
'''
{user_persona}
'''

### Agent Character
'''
{agent_character}
'''

### Dialogue Context
'''
{str_dialogue_context}
'''

"""


def import_template(mode):
    if mode == "thought_completion":
        return system_prompt_for_SThought, base_prompt_for_SThought
    elif mode == "model_pxy":
        return system_prompt_for_model_pxy, base_prompt_for_model_pxy
    else:
        raise ValueError("Invalid choice")
