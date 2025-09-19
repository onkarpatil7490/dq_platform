from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils import (
    chatbot, convert_rule_to_sql,
    delete_rule, get_all_rules_of_table,
    get_info, get_query_test_results,
    get_rule_result_table,
    get_rule_suggestion_on_column,
    get_sql_queries_for_rules, insert_rule,
    load_col_values, load_table_values,
    process_and_save_rule_results,
    process_table_rules_parallel,
    transform_query

)

app = FastAPI(
    title="Data Quality Rule Management API",
    description="APIs to manage data quality rules, validate queries, fetch table/column data, and interact with an AI chatbot.",
    version="1.0.0"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  
    allow_headers=["*"],
    expose_headers=["*"],
)

class ConvertRuleRequest(BaseModel):
    table_name: str = Field(..., description="Name of the database table", example="meter_data")
    column_name: str = Field(..., description="Column on which the rule is applied", example="pincode")
    rule: str = Field(..., description="Rule text to be converted into SQL", example="there should not be a null value")


class AddRuleRequest(BaseModel):
    rule: str = Field(..., description="Rule text to be converted into SQL", example="there should not be a null value")
    table_name: str = Field(..., description="Name of the database table", example="meter_data")
    column_name: str = Field(..., description="Column on which the rule is applied", example="pincode")
    rule_category: str = Field(..., description="Category of rule (e.g., info, warning, error)", example="info")
    sql_query_usr: str = Field(..., description="SQL representation of the rule to display on the UI", example="SELECT pincode FROM meter_data WHERE pincode IS NOT NULL")
    sql_query_val: str = Field(..., description="SQL query to validate the rule", example="SELECT row_num FROM meter_data WHERE pincode IS NOT NULL")


class DeleteRuleRequest(BaseModel):
    rule_id: int = Field(..., description="Number of the rule stored in rule_manage table to be deleted", example=1)


class RuleSuggestionRequest(BaseModel):
    table_name: str = Field(..., description="Name of the database table", example="meter_data")
    column_name: str = Field(..., description="Column on which the rule is applied", example="pincode")
    existing_rules: Optional[List[str]] = Field(default=[], description="List of existing rules for the column", example=["must not be null", "must be unique"])


class TableDataRequest(BaseModel):
    table_name: str = Field(..., description="Name of the database table", example="meter_data")
    offset: int = Field(default=0, description="Row offset (for pagination)", example=0)
    limit: int = Field(default=100, description="Maximum number of rows to return", example=50)


class ColumnDataRequest(BaseModel):
    table_name: str = Field(..., description="Name of the database table", example="meter_data")
    column_name: str = Field(..., description="Column on which the rule is applied", example="pincode")
    offset: int = Field(default=0, description="Row offset (for pagination)", example=0)
    limit: int = Field(description="Maximum number of rows to return", example=50)


class ChatbotRequest(BaseModel):
    user_input: str = Field(..., description="User query or message to the AI chatbot", example="Suggest a rule for validating not null values")
    table_name: str = Field(..., description="Name of the database table", example="meter_data")
    column_name: str = Field(..., description="Column on which the rule is applied", example="pincode")


class ValidateSQLRequest(BaseModel):
    sql_query: str = Field(..., description="SQL representation of the rule", example="SELECT row_num FROM meter_data WHERE pincode IS NOT NULL")
    table_name: str = Field(..., description="Name of the database table", example="meter_data")
    column_name: str = Field(..., description="Column on which the rule is applied", example="pincode")

class InfoRequest(BaseModel):
    table_name: str = Field(..., description="Name of the database table", example="meter_data")
    column_name: str = Field(..., description="Column on which the rule is applied", example="pincode")


# API Endpoints
@app.get("/")
def root():
    return {"message": "Welcome to Data Sentinel application!"}

@app.post("/convert_rule_to_sql/")
def convert_rule_to_sql_api(request: ConvertRuleRequest):
    sql_output = convert_rule_to_sql(request.rule, request.table_name, request.column_name)
    return JSONResponse(content={"sql_output": sql_output})


@app.post("/validate_sql_query/")
def validate_sql_query(request: ValidateSQLRequest):
    validation_sql_query = transform_query(request.sql_query)
    stats_dict = get_query_test_results(validation_sql_query, request.column_name, request.table_name)
    return JSONResponse(content={"stats": stats_dict})


@app.put("/add_rule/")
def add_rule_api(request: AddRuleRequest):
    insert_rule(request.rule, request.table_name, request.column_name, 
                request.rule_category, request.sql_query_usr, request.sql_query_val)
    return JSONResponse(content={"message": f"Rule inserted successfully."})


@app.delete("/delete_rule/")
def delete_rule_api(request: DeleteRuleRequest):
    delete_rule(request.rule_id)
    return JSONResponse(content={"message": f"Rule {request.rule_id} deleted successfully (if it existed)."})


@app.post("/get_rule_suggestion/")
def get_rule_suggestion_api(request: RuleSuggestionRequest):
    suggested_rule = get_rule_suggestion_on_column(request.column_name, request.table_name, request.existing_rules)
    return JSONResponse(content={"suggested_rule": suggested_rule})


@app.get("/get_all_rules_of_table/")
def get_all_rules_of_table_api(table_name: str = Query(..., description="Table name", example="meter_data"),
                               column_name: str = Query(None, description="Column name", example="pincode")):
    rules = get_all_rules_of_table(table_name, column_name)
    return JSONResponse(content={"rules": rules})


@app.post("/get_table_data/")
def get_table_data_api(request: TableDataRequest):
    columns, data = load_table_values(request.table_name, request.offset, request.limit)
    return JSONResponse(content={"columns": columns, "rows": data})


@app.post("/get_col_data/")
def get_col_data_api(request: ColumnDataRequest):
    data = load_col_values(request.table_name, request.column_name, request.offset, request.limit)
    return JSONResponse(content={"rows": data})


@app.post("/chatbot/")
def chatbot_api(request: ChatbotRequest):
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=request.user_input)], "current_column": request.column_name},
        config={"configurable": {"thread_id": "thread_id-1"}}
    )
    return JSONResponse(content={"AI Response": response["messages"][-1].content})


@app.post("/info/")
def info_api(request: InfoRequest):
    info = get_info(request.table_name, request.column_name)
    return JSONResponse(content={"info": info})


@app.post("/process_rules_and_save_result/")
def process_rules_and_save_result():
    rules = get_sql_queries_for_rules(rule_id_list=[])
    result = process_table_rules_parallel(rules)
    process_and_save_rule_results(result)
    return JSONResponse(content={"message": f"Rule results processed and saved successfully."})


@app.get("/get_rule_result_table/")
def get_rule_result():
    rule_result = get_rule_result_table()
    return JSONResponse(content={"rules": rule_result})