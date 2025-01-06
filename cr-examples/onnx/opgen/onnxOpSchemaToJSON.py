import json

from onnx.defs import (
    OpSchema,
    get_all_schemas_with_history,
)

class OpSchemaEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, OpSchema):
            return {
                "file": obj.file,
                "line": obj.line,
                "support_level": obj.support_level.name,
                "doc": obj.doc,
                "since_version": obj.since_version,
                "deprecated": obj.deprecated,
                "domain": obj.domain,
                "name": obj.name,
                "min_input": obj.min_input,
                "max_input": obj.max_input,
                "min_output": obj.min_output,
                "max_output": obj.max_output,
                "attributes": obj.attributes,
                "inputs": obj.inputs,
                "outputs": obj.outputs,
                "type_constraints": obj.type_constraints,
                "has_function": obj.has_function,
                "has_context_dependent_function": obj.has_context_dependent_function,
                "has_data_propagation_function": obj.has_data_propagation_function,
                "has_type_and_shape_inference_function": obj.has_type_and_shape_inference_function,
                # @@@ useful to decode to Java ONNX model and then Java source
                # "function_body": obj.function_body.__str__()
            }
        elif isinstance(obj, OpSchema.FormalParameter):
            return {
                "name": obj.name,
                # @@@ Convert to array of string, but might not be needed, see type_constraints
                "types": obj.types.__str__(),
                "type_str": obj.type_str,
                "description": obj.description,
                "option": obj.option.name,
                "min_arity": obj.min_arity,
                "is_homogeneous": obj.is_homogeneous,
                "differentiation_category": obj.differentiation_category.name,
            }
        elif isinstance(obj, OpSchema.Attribute):
            return {
                "name": obj.name,
                "description": obj.description,
                "type": obj.type.name,
                # @@@ extract default value from protobuf
                "default_value": obj.default_value.__str__(),
                "required": obj.required,
            }
        elif isinstance(obj, OpSchema.TypeConstraintParam):
            return {
                "type_param_str": obj.type_param_str,
                "description": obj.description,
                "allowed_type_strs": obj.allowed_type_strs,
            }
        return super().default(obj)

schemas: list[OpSchema] = get_all_schemas_with_history()

json = json.dumps(schemas, cls=OpSchemaEncoder, indent=4)
print(json)
