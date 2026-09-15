import json

CALCULATOR_TOOL ={
  "type":"function",
  "function":{
    "name":"calculator",
    "description":"Perform basic arithmetic calculations.",
    "parameters":{
      "type":"object",
      "properties":{
        "operation":{
          "type":"string",
          "enum":[
            "add",
            "subtract",
            "multiply",
            "divide"
          ],
          "description":"The arithmetic operation to perform."
        },
        "a":{
          "type":"number",
          "description":"The first number"
        },
        "b":{
          "type":"number",
          "description":"The second number."
        }
      },
      "required":[
        "operation","a","b"
      ]
    }
  }
}

TOOLS = [CALCULATOR_TOOL]

def calculator(operation:str,a:float,b:float) -> float | str:
    if operation == "add":
        return a + b
    
    if operation == "subtract":
        return a - b 
    
    if operation == "multiply":
        return a * b
    
    if operation == "divide":
        if b == 0:
          return "Error: Cannot divide by zero."
        return a / b
    
    return f"Error: Unknow operation {operation}."


def execute_tool(tool_name:str,arguments:str):
  """
  Execute a tool using the arguments returnd by the LLM.
  """
  if tool_name != "calculator":
    return f"Error: Unknow tool '{tool_name}'"
  
  try:
      args = json.loads(arguments)

      return calculator(
        operation=args["operation"],
        a=args["a"],
        b=args["b"]
      )
  
  except(json.JSONDecodeError,KeyError,TypeError) as e:
    return f"Error executing calculator:{str(e)}"