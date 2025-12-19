
import Image from '@theme/IdealImage';


# SCIM with Dheera AI

✨ **Enterprise**: SCIM support requires a premium license.

Enables identity providers (Okta, Azure AD, OneLogin, etc.) to automate user and team (group) provisioning, updates, and deprovisioning on Dheera AI.


This tutorial will walk you through the steps to connect your IDP to Dheera AI SCIM Endpoints.

### Supported SSO Providers for SCIM
Below is a list of supported SSO providers for connecting to Dheera AI SCIM Endpoints.
- Microsoft Entra ID (Azure AD)
- Okta
- Google Workspace
- OneLogin
- Keycloak
- Auth0


## 1. Get your SCIM Tenant URL and Bearer Token

On Dheera AI, navigate to the Settings > Admin Settings > SCIM. On this page you will create a SCIM Token, this allows your IDP to authenticate to dheera_ai `/scim` endpoints.

<Image img={require('../../img/scim_2.png')}  style={{ width: '800px', height: 'auto' }} />

## 2. Connect your IDP to Dheera AI SCIM Endpoints

On your IDP provider, navigate to your SSO application and select `Provisioning` > `New provisioning configuration`.

On this page, paste in your dheera_ai scim tenant url and bearer token.

Once this is pasted in, click on `Test Connection` to ensure your IDP can authenticate to the Dheera AI SCIM endpoints.

<Image img={require('../../img/scim_4.png')}  style={{ width: '800px', height: 'auto' }} />


## 3. Test SCIM Connection

### 3.1 Assign the group to your Dheera AI Enterprise App

On your IDP Portal, navigate to `Enterprise Applications` > Select your dheera_ai app 

<Image img={require('../../img/msft_enterprise_app.png')}  style={{ width: '800px', height: 'auto' }} />

<br />
<br />

Once you've selected your dheera_ai app, click on `Users and Groups` > `Add user/group` 

<Image img={require('../../img/msft_enterprise_assign_group.png')}  style={{ width: '800px', height: 'auto' }} />

<br />

Now select the group you created in step 1.1. And add it to the Dheera AI Enterprise App. At this point we have added `Production LLM Evals Group` to the Dheera AI Enterprise App. The next step is having Dheera AI automatically create the `Production LLM Evals Group` on the Dheera AI DB when a new user signs in.

<Image img={require('../../img/msft_enterprise_select_group.png')}  style={{ width: '800px', height: 'auto' }} />


### 3.2 Sign in to Dheera AI UI via SSO

Sign into the Dheera AI UI via SSO. You should be redirected to the Entra ID SSO page. This SSO sign in flow will trigger Dheera AI to fetch the latest Groups and Members from Azure Entra ID.

<Image img={require('../../img/msft_sso_sign_in.png')}  style={{ width: '800px', height: 'auto' }} />

### 3.3 Check the new team on Dheera AI UI

On the Dheera AI UI, Navigate to `Teams`, You should see the new team `Production LLM Evals Group` auto-created on Dheera AI. 

<Image img={require('../../img/msft_auto_team.png')}  style={{ width: '900px', height: 'auto' }} />

> **Note:** When a user is removed from your organization via SCIM, all API keys and access tokens associated with that user will be automatically deleted from Dheera AI. This ensures that removed users lose all access immediately and securely.



