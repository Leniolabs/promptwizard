from marshmallow import Schema, fields, ValidationError, validates, validates_schema

class ModelSchema(Schema):
        # Schema for defining the GPT model attributes
        name = fields.Str(required=True)
        temperature = fields.Float(required=True)
        max_tokens = fields.Integer(required=True, strict=True)

        @validates('name')
        def validate_name(self, model):
            # Validate the 'name' field against a list of allowed names
            allowed_names = ['gpt-3.5-turbo', 'gpt-4']
            if model not in allowed_names:
                raise ValidationError(f"'name' must be one of the following: {', '.join(allowed_names)}")
            
        @validates("temperature")
        def validate_temperature_range(self, value):
            # Validate the 'temperature' field to be within a specified range
            if not (0 <= value <= 2):
                raise ValidationError("'temperature' must be a float between 0 and 2.")
            
        @validates("max_tokens")
        def validate_max_tokens(self, value):
            # Validate the 'max_tokens' field to be greater than 0
            if not (1 <= value):
                raise ValidationError("'max_tokens' must be an integer greater than 0.")

class EloFormatTestCaseSchema(Schema):
    # Schema for test cases with the 'Elo' method
    cases = fields.List(fields.Str(required=True))
    method = fields.Str(required=True)
    model = fields.Nested(ModelSchema)
    description = fields.Str(required=True)

    @validates("method")
    def validate_method(self, value):
        # Validate the 'method' field to be 'Elo'
        expected_method = "Elo"
        if value != expected_method:
            raise ValidationError(f"Method must be '{expected_method}'.")

class ClaEqInTestCaseSchema(Schema):
    # Schema for individual test cases with 'Classification', 'Equals', and 'Includes' methods
    inout = fields.String(required=True)
    output = fields.String(required=True)

class ClaEqInFormatTestCaseSchema(Schema):
        # Schema for test cases formatted for 'Classification', 'Equals', and 'Includes' methods
        cases = fields.List(fields.Nested(ClaEqInTestCaseSchema), required=True)
        method = fields.Str(required=True)
        model = fields.Nested(ModelSchema)

        @validates("method")
        def validate_method(self, value):
            # Validate the 'method' field to be one of 'Classification', 'Equals', or 'Includes'
            expected_method1 = "Classification"
            expected_method2 = "Equals"
            expected_method3 = "Includes"
            if value != expected_method1 and value != expected_method2 and value != expected_method3:
                raise ValidationError(f"Method must be '{expected_method1}', '{expected_method2}' or '{expected_method3}'.")

class FunctionCallingFormatFunctionSchema(Schema):
        name = fields.Str(required=True)
        description = fields.Str(required=True)
        parameters = fields.Dict(
        properties=fields.Field(required=True)
        
    )

class FunctionCallingTestCaseSchema(Schema):
    # Schema for defining function attributes used in 'function_calling' method
    inout = fields.Str(required=True)
    output1 = fields.Str(required=True)
    output2 = fields.Str(required=True)

class FunctionCallingFormatTestCaseSchema(Schema):
        # Schema for test cases formatted for 'function_calling' method
        cases = fields.List(fields.Nested(FunctionCallingTestCaseSchema))
        method = fields.Str(required=True)
        model = fields.Nested(ModelSchema)
        functions = fields.List(fields.Nested(FunctionCallingFormatFunctionSchema))
        function_call = fields.Str()

        @validates("cases")
        def validate_cases_list(self, cases):
            # Validate that each test case consists of 3 strings: test case, correct function, and correct variable
            for case in cases:
                if len(case) != 3:
                    raise ValidationError("Each test case must consist of 3 strings, the first one being the specific test case, the second one the correct function to use, and the third one the correct variable.")
                
        @validates("method")
        def validate_method(self, value):
            # Validate that the 'method' field is 'function_calling'
            expected_method = "function_calling"
            if value != expected_method:
                raise ValidationError(f"Method must be '{expected_method}'.")
            
class IterationsSchema(Schema):
        # Schema for defining iteration settings
        number = fields.Integer(strict=True, missing=0)
        best_prompts = fields.Integer(strict=True)
        model = fields.Nested(ModelSchema)

        @validates("number")
        def validate_number_iterations(self, value):
            # Validate that 'number' is a non-negative integer
            if not (0 <= value):
                raise ValidationError("'number' must be an integer equal or greater than 0.")

class GenerationModelSchema(Schema):
        # Schema for defining generation settings
        model = fields.Nested(ModelSchema)
        number = fields.Integer(missing=4) # Default number of prompts is 4
        constraints = fields.Str()
        description = fields.Str(required=True)

        @validates("number")
        def validate_number_iterations(self, value):
            # Validate that 'number' is an integer equal to or greater than 4
            if not (value >= 4):
                raise ValidationError("'number' must be an integer equal or greater than 4.")

class PromptSchema(Schema):
        # Schema for defining prompts
        list = fields.List(fields.Str())
        iterations = fields.Nested(IterationsSchema)
        generation = fields.Nested(GenerationModelSchema)

        @validates("list")
        def validate_my_list_length(self, value):
            # Validate that the prompt list has at least 4 elements
            if value is not None and len(value) < 4:
                raise ValidationError("The list should have at least 4 elements.")
        


class ValidationFunctionCalling(Schema):
        # Schema for validating 'function_calling' test cases and prompts
        test = fields.Nested(FunctionCallingFormatTestCaseSchema, required=True)
        prompts = fields.Nested(PromptSchema, required=True)

class ValidationClaEqIn(Schema):
        # Schema for validating 'Classification', 'Equals', and 'Includes' test cases and prompts
        test = fields.Nested(ClaEqInFormatTestCaseSchema, required=True)
        prompts = fields.Nested(PromptSchema, required=True)

class ValidationElo(Schema):
        # Schema for validating 'Elo' test cases and prompts
        test = fields.Nested(EloFormatTestCaseSchema, required=True)
        prompts = fields.Nested(PromptSchema, required=True)
