# What is stored in the DB

The Dheera AI Proxy uses a PostgreSQL database to store various information. Here's are the main features the DB is used for:
- Virtual Keys, Organizations, Teams, Users, Budgets, and more.
- Per request Usage Tracking

## Link to DB Schema

You can see the full DB Schema [here](https://github.com/BerriAI/dheera_ai/blob/main/schema.prisma)

## DB Tables

### Organizations, Teams, Users, End Users

| Table Name | Description | Row Insert Frequency |
|------------|-------------|---------------------|
| Dheera AI_OrganizationTable | Manages organization-level configurations. Tracks organization spend, model access, and metadata. Links to budget configurations and teams. | Low |
| Dheera AI_TeamTable | Handles team-level settings within organizations. Manages team members, admins, and their roles. Controls team-specific budgets, rate limits, and model access. | Low |
| Dheera AI_UserTable | Stores user information and their settings. Tracks individual user spend, model access, and rate limits. Manages user roles and team memberships. | Low |
| Dheera AI_EndUserTable | Manages end-user configurations. Controls model access and regional requirements. Tracks end-user spend. | Low |
| Dheera AI_TeamMembership | Tracks user participation in teams. Manages team-specific user budgets and spend. | Low |
| Dheera AI_OrganizationMembership | Manages user roles within organizations. Tracks organization-specific user permissions and spend. | Low |
| Dheera AI_InvitationLink | Handles user invitations. Manages invitation status and expiration. Tracks who created and accepted invitations. | Low |
| Dheera AI_UserNotifications | Handles model access requests. Tracks user requests for model access. Manages approval status. | Low |

### Authentication

| Table Name | Description | Row Insert Frequency |
|------------|-------------|---------------------|
| Dheera AI_VerificationToken | Manages Virtual Keys and their permissions. Controls token-specific budgets, rate limits, and model access. Tracks key-specific spend and metadata. | **Medium** - stores all Virtual Keys |

### Model (LLM) Management

| Table Name | Description | Row Insert Frequency |
|------------|-------------|---------------------|
| Dheera AI_ProxyModelTable | Stores model configurations. Defines available models and their parameters. Contains model-specific information and settings. | Low - Configuration only |

### Budget Management

| Table Name | Description | Row Insert Frequency |
|------------|-------------|---------------------|
| Dheera AI_BudgetTable | Stores budget and rate limit configurations for organizations, keys, and end users. Tracks max budgets, soft budgets, TPM/RPM limits, and model-specific budgets. Handles budget duration and reset timing. | Low - Configuration only |


### Tracking & Logging

| Table Name | Description | Row Insert Frequency |
|------------|-------------|---------------------|
| Dheera AI_SpendLogs | Detailed logs of all API requests. Records token usage, spend, and timing information. Tracks which models and keys were used. | **Medium - this is a batch process that runs on an interval.** |
| Dheera AI_AuditLog | Tracks changes to system configuration. Records who made changes and what was modified. Maintains history of updates to teams, users, and models. | **Off by default**, **High - Runs on every change to an entity** |

## Disable `Dheera AI_SpendLogs`

You can disable spend_logs and error_logs by setting `disable_spend_logs` and `disable_error_logs` to `True` on the `general_settings` section of your proxy_config.yaml file.

```yaml
general_settings:
  disable_spend_logs: True   # Disable writing spend logs to DB
  disable_error_logs: True   # Only disable writing error logs to DB, regular spend logs will still be written unless `disable_spend_logs: True`
```

### What is the impact of disabling these logs?

When disabling spend logs (`disable_spend_logs: True`):
- You **will not** be able to view Usage on the Dheera AI UI
- You **will** continue seeing cost metrics on s3, Prometheus, Langfuse (any other Logging integration you are using)

When disabling error logs (`disable_error_logs: True`):
- You **will not** be able to view Errors on the Dheera AI UI
- You **will** continue seeing error logs in your application logs and any other logging integrations you are using


## Migrating Databases 

If you need to migrate Databases the following Tables should be copied to ensure continuation of services and no downtime


| Table Name | Description | 
|------------|-------------|
| Dheera AI_VerificationToken | **Required** to ensure existing virtual keys continue working |
| Dheera AI_UserTable | **Required** to ensure existing virtual keys continue working |
| Dheera AI_TeamTable | **Required** to ensure Teams are migrated |
| Dheera AI_TeamMembership | **Required** to ensure Teams member budgets are migrated |
| Dheera AI_BudgetTable | **Required** to migrate existing budgeting settings |
| Dheera AI_OrganizationTable | **Optional** Only migrate if you use Organizations in DB |
| Dheera AI_OrganizationMembership | **Optional** Only migrate if you use Organizations in DB | 
| Dheera AI_ProxyModelTable | **Optional** Only migrate if you store your LLMs in the DB (i.e you set `STORE_MODEL_IN_DB=True`) |
| Dheera AI_SpendLogs | **Optional** Only migrate if you want historical data on Dheera AI UI |
| Dheera AI_ErrorLogs | **Optional** Only migrate if you want historical data on Dheera AI UI |


