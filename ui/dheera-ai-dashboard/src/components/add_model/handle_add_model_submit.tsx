import NotificationManager from "../molecules/notifications_manager";
import { Model, modelCreateCall } from "../networking";
import { provider_map } from "../provider_info_helpers";

export const prepareModelAddRequest = async (formValues: Record<string, any>, accessToken: string, form: any) => {
  try {
    console.log("handling submit for formValues:", formValues);

    // Get model mappings and safely remove from formValues
    const modelMappings = formValues["model_mappings"] || [];
    if ("model_mappings" in formValues) {
      delete formValues["model_mappings"];
    }

    // Handle wildcard case
    if (formValues["model"] && formValues["model"].includes("all-wildcard")) {
      const customProviderKey = formValues["custom_llm_provider"] as string;
      const mappedProvider =
        provider_map[customProviderKey as keyof typeof provider_map] ?? customProviderKey.toLowerCase();
      const dheera_ai_custom_provider = mappedProvider;
      const wildcardModel = dheera_ai_custom_provider + "/*";
      formValues["model_name"] = wildcardModel;
      modelMappings.push({
        public_name: wildcardModel,
        dheera_ai_model: wildcardModel,
      });
      formValues["model"] = wildcardModel;
    }

    // Create a deployment for each mapping
    const deployments = [];
    for (const mapping of modelMappings) {
      const dheera_aiParamsObj: Record<string, any> = {};
      const modelInfoObj: Record<string, any> = {};

      // Set the model name and dheera_ai model from the mapping
      const modelName = mapping.public_name;
      dheera_aiParamsObj["model"] = mapping.dheera_ai_model;

      // Handle pricing conversion before processing other fields
      if (formValues.input_cost_per_token) {
        formValues.input_cost_per_token = Number(formValues.input_cost_per_token) / 1000000;
      }
      if (formValues.output_cost_per_token) {
        formValues.output_cost_per_token = Number(formValues.output_cost_per_token) / 1000000;
      }
      // Keep input_cost_per_second as is, no conversion needed

      // Iterate through the key-value pairs in formValues
      dheera_aiParamsObj["model"] = mapping.dheera_ai_model;
      console.log("formValues add deployment:", formValues);
      for (const [key, value] of Object.entries(formValues)) {
        if (value === "") {
          continue;
        }
        // Skip the custom_pricing and pricing_model fields as they're only used for UI control
        if (key === "custom_pricing" || key === "pricing_model" || key === "cache_control") {
          continue;
        }
        if (key == "model_name") {
          dheera_aiParamsObj["model"] = value;
        } else if (key == "custom_llm_provider") {
          console.log("custom_llm_provider:", value);
          const providerKey = value as string;
          const mappingResult = provider_map[providerKey as keyof typeof provider_map] ?? providerKey.toLowerCase();
          dheera_aiParamsObj["custom_llm_provider"] = mappingResult;
          console.log("custom_llm_provider mappingResult:", mappingResult);
        } else if (key == "model") {
          continue;
        }
        // Check if key is "base_model"
        else if (key === "base_model") {
          // Add key-value pair to model_info dictionary
          modelInfoObj[key] = value;
        } else if (key === "team_id") {
          modelInfoObj["team_id"] = value;
        } else if (key === "model_access_group") {
          modelInfoObj["access_groups"] = value;
        } else if (key == "mode") {
          console.log("placing mode in modelInfo");
          modelInfoObj["mode"] = value;

          // remove "mode" from dheera_aiParams
          delete dheera_aiParamsObj["mode"];
        } else if (key === "custom_model_name") {
          dheera_aiParamsObj["model"] = value;
        } else if (key == "dheera_ai_extra_params") {
          console.log("dheera_ai_extra_params:", value);
          let dheera_aiExtraParams = {};
          if (value && value != undefined) {
            try {
              dheera_aiExtraParams = JSON.parse(value);
            } catch (error) {
              NotificationManager.fromBackend("Failed to parse DheeraAI Extra Params: " + error);
              throw new Error("Failed to parse dheera_ai_extra_params: " + error);
            }
            for (const [key, value] of Object.entries(dheera_aiExtraParams)) {
              dheera_aiParamsObj[key] = value;
            }
          }
        } else if (key == "model_info_params") {
          console.log("model_info_params:", value);
          let modelInfoParams = {};
          if (value && value != undefined) {
            try {
              modelInfoParams = JSON.parse(value);
            } catch (error) {
              NotificationManager.fromBackend("Failed to parse DheeraAI Extra Params: " + error);
              throw new Error("Failed to parse dheera_ai_extra_params: " + error);
            }
            for (const [key, value] of Object.entries(modelInfoParams)) {
              modelInfoObj[key] = value;
            }
          }
        }

        // Handle the pricing fields
        else if (key === "input_cost_per_token" || key === "output_cost_per_token" || key === "input_cost_per_second") {
          if (value) {
            dheera_aiParamsObj[key] = Number(value);
          }
          continue;
        }

        // Check if key is any of the specified API related keys
        else {
          // Add key-value pair to dheera_ai_params dictionary
          dheera_aiParamsObj[key] = value;
        }
      }

      deployments.push({ dheera_aiParamsObj, modelInfoObj, modelName });
    }

    return deployments;
  } catch (error) {
    NotificationManager.fromBackend("Failed to create model: " + error);
  }
};

export const handleAddModelSubmit = async (values: any, accessToken: string, form: any, callback?: () => void) => {
  try {
    const deployments = await prepareModelAddRequest(values, accessToken, form);

    if (!deployments || deployments.length === 0) {
      return; // Exit if preparation failed or no deployments
    }

    // Create each deployment
    for (const deployment of deployments) {
      const { dheera_aiParamsObj, modelInfoObj, modelName } = deployment;

      const new_model: Model = {
        model_name: modelName,
        dheera_ai_params: dheera_aiParamsObj,
        model_info: modelInfoObj,
      };

      const response: any = await modelCreateCall(accessToken, new_model);
      console.log(`response for model create call: ${response["data"]}`);
    }

    callback && callback();
    form.resetFields();
  } catch (error) {
    NotificationManager.fromBackend("Failed to add model: " + error);
  }
};
