import Image from '@theme/IdealImage';

# Hashicorp Vault

:::info

✨ **This is an Enterprise Feature**

[Enterprise Pricing](https://www.dheera_ai.ai/#pricing)

[Contact us here to get a free trial](https://calendly.com/d/4mp-gd3-k5k/dheera_ai-1-1-onboarding-chat)

:::

| Feature | Support | Description |
|---------|----------|-------------|
| Reading Secrets | ✅ | Read secrets e.g `OPENAI_API_KEY` |
| Writing Secrets | ✅ | Store secrets e.g `Virtual Keys` |
| Authentication Methods to Hashicorp Vault | ✅ | AppRole, TLS Certificate, Token |

Read secrets from [Hashicorp Vault](https://developer.hashicorp.com/vault/docs/secrets/kv/kv-v2)

**Step 1.** Add Hashicorp Vault details in your environment

Dheera AI supports three methods of authentication:

1. AppRole authentication (recommended) - `HCP_VAULT_APPROLE_ROLE_ID` and `HCP_VAULT_APPROLE_SECRET_ID`
2. TLS cert authentication - `HCP_VAULT_CLIENT_CERT` and `HCP_VAULT_CLIENT_KEY`
3. Token authentication - `HCP_VAULT_TOKEN`

```bash
HCP_VAULT_ADDR="https://test-cluster-public-vault-0f98180c.e98296b2.z1.hashicorp.cloud:8200"
HCP_VAULT_NAMESPACE="admin"

# Authentication via AppRole (recommended)
HCP_VAULT_APPROLE_ROLE_ID="your-role-id"
HCP_VAULT_APPROLE_SECRET_ID="your-secret-id"
HCP_VAULT_APPROLE_MOUNT_PATH="approle" # OPTIONAL. defaults to "approle"

# OR - Authentication via TLS cert
HCP_VAULT_CLIENT_CERT="path/to/client.pem"
HCP_VAULT_CLIENT_KEY="path/to/client.key"

# OR - Authentication via token
HCP_VAULT_TOKEN="hvs.CAESIG52gL6ljBSdmq*****"


# OPTIONAL
HCP_VAULT_REFRESH_INTERVAL="86400" # defaults to 86400, frequency of cache refresh for Hashicorp Vault
HCP_VAULT_MOUNT_NAME="secret" # OPTIONAL. defaults to "secret", set this if your KV engine is mounted elsewhere
HCP_VAULT_PATH_PREFIX="dheera_ai" # OPTIONAL. defaults to None, set this if your secrets live under a custom prefix like secret/data/dheera_ai/OPENAI_API_KEY
```

**Step 2.** Add to proxy config.yaml

```yaml
general_settings:
  key_management_system: "hashicorp_vault"

  # [OPTIONAL SETTINGS]
  key_management_settings: 
    store_virtual_keys: true # OPTIONAL. Defaults to False, when True will store virtual keys in secret manager
    prefix_for_stored_virtual_keys: "dheera_ai/" # OPTIONAL. If set, this prefix will be used for stored virtual keys in the secret manager
    access_mode: "read_and_write" # Literal["read_only", "write_only", "read_and_write"]
```

**Step 3.** Start + test proxy

```
$ dheera_ai --config /path/to/config.yaml
```

[Quick Test Proxy](../proxy/user_keys)


## Authentication Methods

Dheera AI supports three authentication methods for Hashicorp Vault, with the following priority:

1. **AppRole** - Recommended for production applications
2. **TLS Certificate** - For certificate-based authentication
3. **Token** - Direct token authentication

### 1. AppRole Authentication

To set up AppRole authentication:

1. Enable AppRole auth in Vault:
```bash
vault auth enable approle
```

2. Create a policy and role for Dheera AI:
```bash
# Create a policy file (dheera_ai-policy.hcl)
path "secret/data/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Apply the policy
vault policy write dheera_ai-policy dheera_ai-policy.hcl

# Create an AppRole
vault write auth/approle/role/dheera_ai \
    token_policies="dheera_ai-policy" \
    token_ttl=32d \
    token_max_ttl=32d
```

3. Get your Role ID and Secret ID:
```bash
# Get Role ID
vault read auth/approle/role/dheera_ai/role-id

# Generate Secret ID
vault write -f auth/approle/role/dheera_ai/secret-id
```

4. Set the environment variables:
```bash
export HCP_VAULT_APPROLE_ROLE_ID="your-role-id"
export HCP_VAULT_APPROLE_SECRET_ID="your-secret-id"
```

### 2. TLS Certificate Authentication

TLS Certificate authentication uses client certificates for mutual TLS authentication with Vault.

**Environment Variables:**
```bash
export HCP_VAULT_CLIENT_CERT="path/to/client.pem"
export HCP_VAULT_CLIENT_KEY="path/to/client.key"
export HCP_VAULT_CERT_ROLE="your-cert-role"  # Optional
```

**How it works:**
- Dheera AI uses the client certificate and key for mutual TLS authentication
- Vault validates the certificate and issues a temporary token
- The token is cached for the duration of its lease

### 3. Token Authentication

Direct token authentication uses a static Vault token.

**Environment Variables:**
```bash
export HCP_VAULT_TOKEN="hvs.CAESIG52gL6ljBSdmq*****"
```

## How it works

**Reading Secrets**

Dheera AI reads secrets from Hashicorp Vault's KV v2 engine using the following URL format:
```
{VAULT_ADDR}/v1/{NAMESPACE}/{MOUNT_NAME}/data/{PATH_PREFIX}/{SECRET_NAME}
```

For example, if you have:
- `HCP_VAULT_ADDR="https://vault.example.com:8200"`
- `HCP_VAULT_NAMESPACE="admin"`
- `HCP_VAULT_MOUNT_NAME="secret"`
- `HCP_VAULT_PATH_PREFIX="dheera_ai"`
- Secret name: `AZURE_API_KEY`


Dheera AI will look up:
```
https://vault.example.com:8200/v1/admin/secret/data/dheera_ai/AZURE_API_KEY
```

### Expected Secret Format

Dheera AI expects all secrets to be stored as a JSON object with a `key` field containing the secret value.

For example, for `AZURE_API_KEY`, the secret should be stored as:

```json
{
  "key": "sk-1234"
}
```

<Image img={require('../../img/hcorp.png')} />

**Writing Secrets**

When a Virtual Key is Created / Deleted on Dheera AI, Dheera AI will automatically create / delete the secret in Hashicorp Vault.

- Create Virtual Key on Dheera AI either through the Dheera AI Admin UI or API

<Image img={require('../../img/hcorp_create_virtual_key.png')} />


- Check Hashicorp Vault for secret

Dheera AI stores secret under the `prefix_for_stored_virtual_keys` path (default: `dheera_ai/`)

<Image img={require('../../img/hcorp_virtual_key.png')} />
