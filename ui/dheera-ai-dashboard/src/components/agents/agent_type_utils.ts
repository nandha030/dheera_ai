import { Agent } from "./types";
import { AgentCreateInfo } from "../networking";

/**
 * Detects the agent type from an agent's dheera_ai_params.
 * Returns the agent_type string (e.g., "langgraph", "azure_ai_foundry", "bedrock_agentcore", or "a2a")
 */
export const detectAgentType = (agent: Agent): string => {
  const model = agent.dheera_ai_params?.model || "";
  const customProvider = agent.dheera_ai_params?.custom_llm_provider;

  // Check by custom_llm_provider first
  if (customProvider === "langgraph") return "langgraph";
  if (customProvider === "azure_ai") return "azure_ai_foundry";
  if (customProvider === "bedrock") return "bedrock_agentcore";

  // Check by model prefix
  if (model.startsWith("langgraph/")) return "langgraph";
  if (model.startsWith("azure_ai/agents/")) return "azure_ai_foundry";
  if (model.startsWith("bedrock/agentcore/")) return "bedrock_agentcore";

  // Default to a2a
  return "a2a";
};

/**
 * Parses agent data for dynamic form fields (non-A2A agents).
 * Extracts values from dheera_ai_params based on the agent type metadata.
 */
export const parseDynamicAgentForForm = (
  agent: Agent,
  agentTypeInfo: AgentCreateInfo
): Record<string, any> => {
  const values: Record<string, any> = {
    agent_name: agent.agent_name,
    description: agent.agent_card_params?.description || "",
  };

  // Extract credential field values from dheera_ai_params
  for (const field of agentTypeInfo.credential_fields) {
    if (field.include_in_dheera_ai_params !== false) {
      values[field.key] = agent.dheera_ai_params?.[field.key] || field.default_value || "";
    } else {
      // For fields not in dheera_ai_params (like agent_id), try to extract from model string
      if (agentTypeInfo.model_template && agent.dheera_ai_params?.model) {
        const model = agent.dheera_ai_params.model;
        const templateParts = agentTypeInfo.model_template.split("/");
        const modelParts = model.split("/");
        
        // Find the placeholder position and extract the value
        templateParts.forEach((part, index) => {
          if (part === `{${field.key}}` && modelParts[index]) {
            values[field.key] = modelParts[index];
          }
        });
      }
    }
  }

  // Extract cost configuration
  values.cost_per_query = agent.dheera_ai_params?.cost_per_query;
  values.input_cost_per_token = agent.dheera_ai_params?.input_cost_per_token;
  values.output_cost_per_token = agent.dheera_ai_params?.output_cost_per_token;

  return values;
};

