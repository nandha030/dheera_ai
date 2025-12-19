# AWS Key Management V1

:::info

✨ **This is an Enterprise Feature**

[Enterprise Pricing](https://www.dheera_ai.ai/#pricing)

[Contact us here to get a free trial](https://calendly.com/d/4mp-gd3-k5k/dheera_ai-1-1-onboarding-chat)

:::

:::tip

[BETA] AWS Key Management v2 is on the enterprise tier. Go [here for docs](../proxy/enterprise.md#beta-aws-key-manager---key-decryption)

:::

Use AWS KMS to storing a hashed copy of your Proxy Master Key in the environment. 

```bash
export DHEERA_AI_MASTER_KEY="djZ9xjVaZ..." # 👈 ENCRYPTED KEY
export AWS_REGION_NAME="us-west-2"
```

```yaml
general_settings:
  key_management_system: "aws_kms"
  key_management_settings:
    hosted_keys: ["DHEERA_AI_MASTER_KEY"] # 👈 WHICH KEYS ARE STORED ON KMS
```

[**See Decryption Code**](https://github.com/BerriAI/dheera_ai/blob/a2da2a8f168d45648b61279d4795d647d94f90c9/dheera_ai/utils.py#L10182)

