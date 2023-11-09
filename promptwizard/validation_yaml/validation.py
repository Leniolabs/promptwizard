from marshmallow import Schema, fields, ValidationError, validates

class ModelSchema(Schema):
        # Schema for defining the GPT model attributes
        name = fields.Str(required=True)
        temperature = fields.Float(required=True)
        max_tokens = fields.Integer(required=True, strict=True)

        @validates('name')
        def validate_name(self, model):
            # Validate the 'name' field against a list of allowed names
            allowed_names = ['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-instruct']
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
        
class CodeTestCaseSchema(Schema):

    input = fields.String(required=True)
    arguments = fields.Raw(required=True)
    output = fields.Raw(required=True)

class ClaEqInTestCaseSchema(Schema):
    # Schema for individual test cases with 'Classification', 'Equals', 'Includes' and 'LogProbs' methods
    input = fields.String(required=True)
    output = fields.String(required=True)

class JSONTestCaseSchema(Schema):

    input = fields.String(required=True)
    output = fields.Dict(required=True)

class EmbeddingsTestCaseSchema(Schema):

    input = fields.String(required=True)
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
                raise ValidationError(f"Method must be '{expected_method1}', '{expected_method2}' '{expected_method3}'.")

class CodeFormatTestCaseSchema(Schema):
        cases = fields.List(fields.Nested(CodeTestCaseSchema), required=True)
        method = fields.Str(required=True)
        model = fields.Nested(ModelSchema)

        @validates("method")
        def validate_method(self, value):
            expected_methods = ['Code Generation']
            if value not in expected_methods:
                raise ValidationError(f"Method must be one of those '{expected_methods}'.")
            
class JSONFormatTestCaseSchema(Schema):
        cases = fields.List(fields.Nested(JSONTestCaseSchema), required=True)
        method = fields.Str(required=True)
        model = fields.Nested(ModelSchema)

class LogProbsFormatTestCaseSchema(Schema):
    # Schema for test cases with the 'LogProbs' method
    cases = fields.List(fields.Nested(ClaEqInTestCaseSchema), required=True)
    method = fields.Str(required=True)
    model = fields.Nested(ModelSchema)

    @validates("method")
    def validate_method(self, value):
        # Validate the 'method' field to be 'LogProbs'
        expected_method = "LogProbs"
        if value != expected_method:
            raise ValidationError(f"Method must be '{expected_method}'.")

class EmbeddingsModel(Schema):
     model_name = fields.Str()
     @validates("model_name")
     def validate_method(self, value):
            expected_methods = ['text-embedding-ada-002', 'text-similarity-ada-001', 'text-search-ada-query-001', 'code-search-ada-code-001', 'code-search-ada-text-001', 'text-similarity-babbage-001', 'text-search-babbage-doc-001', 'text-search-babbage-query-001', 'code-search-babbage-code-001', 'code-search-babbage-text-001', 'text-similarity-curie-001', 'text-search-curie-doc-001', 'text-search-curie-query-001', 'text-similarity-davinci-001', 'text-search-davinci-doc-001', 'text-search-davinci-query-001']
            if value not in expected_methods:
                raise ValidationError(f"Method must be one of those '{expected_methods}'.")

class EmbeddingsFormatTestCaseSchema(Schema):
    cases = fields.List(fields.Nested(EmbeddingsTestCaseSchema), required=True)
    method = fields.Str(required=True)
    model = fields.Nested(ModelSchema)
    embeddings = fields.Nested(EmbeddingsModel)

class FunctionCallingFormatFunctionSchema(Schema):
        name = fields.Str(required=True)
        description = fields.Str(required=True)
        parameters = fields.Dict(
        properties=fields.Field(required=True)
        
    )

class FunctionCallingTestCaseSchema(Schema):
    # Schema for defining function attributes used in 'Function Calling' method
    input = fields.Str(required=True)
    output1 = fields.Str(required=True)
    output2 = fields.Str(required=True)

class FunctionCallingFormatTestCaseSchema(Schema):
        # Schema for test cases formatted for 'Function Calling' method
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
            # Validate that the 'method' field is 'Function Calling'
            expected_method = "Function Calling"
            if value != expected_method:
                raise ValidationError(f"Method must be '{expected_method}'.")
            
class IterationsSchema(Schema):
        # Schema for defining iteration settings
        number = fields.Integer(strict=True, required=True)
        model = fields.Nested(ModelSchema)
        best_percentage = fields.Float()

        @validates("number")
        def validate_number_iterations(self, value):
            # Validate that 'number' is a non-negative integer
            if not (0 <= value):
                raise ValidationError("'number' must be an integer equal or greater than 0.")
        
        @validates("best_percentage")
        def validate_best_percentage(self, value):
            # Validate that 'best_percentage' is a float between 0 and 100 (inclusive)
            if not (0 < value <= 100):
                raise ValidationError("'best_percentage' must be a float stricter than 0 and less than or equal to 100.")

class GenerationModelEloSchema(Schema):
        # Schema for defining generation settings
        model = fields.Nested(ModelSchema)
        number = fields.Integer(missing=4) # Default number of prompts is 4
        constraints = fields.Str()
        description = fields.Str()

        @validates("number")
        def validate_number_iterations(self, value):
            # Validate that 'number' is an integer equal to or greater than 4
            if not (value >= 4):
                raise ValidationError("'number' must be an integer equal or greater than 4.")

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
        best_prompts = fields.Integer(strict=True)

        @validates("list")
        def validate_my_list_length(self, value):
            # Validate that the prompt list has at least 4 elements
            if value is not None and len(value) < 4:
                raise ValidationError("The list should have at least 4 elements.")

class PromptEloSchema(Schema):
        # Schema for defining prompts
        list = fields.List(fields.Str())
        iterations = fields.Nested(IterationsSchema)
        generation = fields.Nested(GenerationModelEloSchema)

        @validates("list")
        def validate_my_list_length(self, value):
            # Validate that the prompt list has at least 4 elements
            if value is not None and len(value) < 4:
                raise ValidationError("The list should have at least 4 elements.")
        


class ValidationFunctionCalling(Schema):
        # Schema for validating 'Function Calling' test cases and prompts
        test = fields.Nested(FunctionCallingFormatTestCaseSchema, required=True)
        prompts = fields.Nested(PromptSchema, required=True)
        n_retries = fields.Integer()
        timeout = fields.Integer()

class ValidationClaEqIn(Schema):
        # Schema for validating 'Classification', 'Equals', 'Includes' and 'LogProbs' test cases and prompts
        test = fields.Nested(ClaEqInFormatTestCaseSchema, required=True)
        prompts = fields.Nested(PromptSchema, required=True)
        n_retries = fields.Integer()
        timeout = fields.Integer()

class ValidationElo(Schema):
        # Schema for validating 'Elo' test cases and prompts
        test = fields.Nested(EloFormatTestCaseSchema, required=True)
        prompts = fields.Nested(PromptEloSchema, required=True)
        n_retries = fields.Integer()
        timeout = fields.Integer()

class ValidationCode(Schema):
        # Schema for validating 'Code Generation' test cases and prompts
        test = fields.Nested(CodeFormatTestCaseSchema, required=True)
        prompts = fields.Nested(PromptSchema, required=True)
        n_retries = fields.Integer()
        timeout = fields.Integer()

class ValidationJSON(Schema):
    # Schema for validating 'JSON Validation' test cases and prompts
    test = fields.Nested(JSONFormatTestCaseSchema, required=True)
    prompts = fields.Nested(PromptSchema, required=True)
    n_retries = fields.Integer()
    timeout = fields.Integer()

class ValidationEmbeddings(Schema):
    # Schema for validating 'semantic_validation' test cases and prompts
    test = fields.Nested(EmbeddingsFormatTestCaseSchema, required=True)
    prompts = fields.Nested(PromptSchema, required=True)
    n_retries = fields.Integer()
    timeout = fields.Integer()

class ValidationLogProbs(Schema):
    test = fields.Nested(LogProbsFormatTestCaseSchema, required=True)
    prompts = fields.Nested(PromptSchema, required=True)
    n_retries = fields.Integer()
    timeout = fields.Integer()