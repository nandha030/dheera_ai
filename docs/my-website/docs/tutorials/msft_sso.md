import Image from '@theme/IdealImage';

# Microsoft SSO: Sync Groups, Members with Dheera AI

Sync Microsoft SSO Groups, Members with Dheera AI Teams. 

<Image img={require('../../img/dheera_ai_entra_id.png')}  style={{ width: '800px', height: 'auto' }} />

<br />
<br />


## Prerequisites

- An Azure Entra ID account with administrative access
- A Dheera AI Enterprise App set up in your Azure Portal
- Access to Microsoft Entra ID (Azure AD)


## Overview of this tutorial

1. Auto-Create Entra ID Groups on Dheera AI Teams 
2. Sync Entra ID Team Memberships
3. Set default params for new teams and users auto-created on Dheera AI

## 1. Auto-Create Entra ID Groups on Dheera AI Teams 

In this step, our goal is to have Dheera AI automatically create a new team on the Dheera AI DB when there is a new Group Added to the Dheera AI Enterprise App on Azure Entra ID.

### 1.1 Create a new group in Entra ID


Navigate to [your Azure Portal](https://portal.azure.com/) > Groups > New Group. Create a new group. 

<Image img={require('../../img/entra_create_team.png')}  style={{ width: '800px', height: 'auto' }} />

### 1.2 Assign the group to your Dheera AI Enterprise App

On your Azure Portal, navigate to `Enterprise Applications` > Select your dheera_ai app 

<Image img={require('../../img/msft_enterprise_app.png')}  style={{ width: '800px', height: 'auto' }} />

<br />
<br />

Once you've selected your dheera_ai app, click on `Users and Groups` > `Add user/group` 

<Image img={require('../../img/msft_enterprise_assign_group.png')}  style={{ width: '800px', height: 'auto' }} />

<br />

Now select the group you created in step 1.1. And add it to the Dheera AI Enterprise App. At this point we have added `Production LLM Evals Group` to the Dheera AI Enterprise App. The next steps is having Dheera AI automatically create the `Production LLM Evals Group` on the Dheera AI DB when a new user signs in.

<Image img={require('../../img/msft_enterprise_select_group.png')}  style={{ width: '800px', height: 'auto' }} />


### 1.3 Sign in to Dheera AI UI via SSO

Sign into the Dheera AI UI via SSO. You should be redirected to the Entra ID SSO page. This SSO sign in flow will trigger Dheera AI to fetch the latest Groups and Members from Azure Entra ID.

<Image img={require('../../img/msft_sso_sign_in.png')}  style={{ width: '800px', height: 'auto' }} />

### 1.4 Check the new team on Dheera AI UI

On the Dheera AI UI, Navigate to `Teams`, You should see the new team `Production LLM Evals Group` auto-created on Dheera AI. 

<Image img={require('../../img/msft_auto_team.png')}  style={{ width: '900px', height: 'auto' }} />

#### How this works

When a SSO user signs in to Dheera AI:
- Dheera AI automatically fetches the Groups under the Dheera AI Enterprise App
- It finds the Production LLM Evals Group assigned to the Dheera AI Enterprise App
- Dheera AI checks if this group's ID exists in the Dheera AI Teams Table
- Since the ID doesn't exist, Dheera AI automatically creates a new team with:
  - Name: Production LLM Evals Group
  - ID: Same as the Entra ID group's ID

## 2. Sync Entra ID Team Memberships

In this step, we will have Dheera AI automatically add a user to the `Production LLM Evals` Team on the Dheera AI DB when a new user is added to the `Production LLM Evals` Group in Entra ID.

### 2.1 Navigate to the `Production LLM Evals` Group in Entra ID

Navigate to the `Production LLM Evals` Group in Entra ID.

<Image img={require('../../img/msft_member_1.png')}  style={{ width: '800px', height: 'auto' }} />


### 2.2 Add a member to the group in Entra ID

Select `Members` > `Add members`

In this stage you should add the user you want to add to the `Production LLM Evals` Team.

<Image img={require('../../img/msft_member_2.png')}  style={{ width: '800px', height: 'auto' }} />



### 2.3 Sign in as the new user on Dheera AI UI

Sign in as the new user on Dheera AI UI. You should be redirected to the Entra ID SSO page. This SSO sign in flow will trigger Dheera AI to fetch the latest Groups and Members from Azure Entra ID. During this step Dheera AI sync it's teams, team members with what is available from Entra ID

<Image img={require('../../img/msft_sso_sign_in.png')}  style={{ width: '800px', height: 'auto' }} />



### 2.4 Check the team membership on Dheera AI UI

On the Dheera AI UI, Navigate to `Teams`, You should see the new team `Production LLM Evals Group`. Since your are now a member of the `Production LLM Evals Group` in Entra ID, you should see the new team `Production LLM Evals Group` on the Dheera AI UI.

<Image img={require('../../img/msft_member_3.png')}  style={{ width: '900px', height: 'auto' }} />

## 3. Set default params for new teams auto-created on Dheera AI

Since dheera_ai auto creates a new team on the Dheera AI DB when there is a new Group Added to the Dheera AI Enterprise App on Azure Entra ID, we can set default params for new teams created. 

This allows you to set a default budget, models, etc for new teams created. 

### 3.1 Set `default_team_params` on dheera_ai 

Navigate to your dheera_ai config file and set the following params 

```yaml showLineNumbers title="dheera_ai config with default_team_params"
dheera_ai_settings:
  default_team_params:             # Default Params to apply when dheera_ai auto creates a team from SSO IDP provider
    max_budget: 100                # Optional[float], optional): $100 budget for the team
    budget_duration: 30d           # Optional[str], optional): 30 days budget_duration for the team
    models: ["gpt-3.5-turbo"]      # Optional[List[str]], optional): models to be used by the team
```

### 3.2 Auto-create a new team on Dheera AI

- In this step you should add a new group to the Dheera AI Enterprise App on Azure Entra ID (like we did in step 1.1). We will call this group `Default Dheera AI Prod Team` on Azure Entra ID.
- Start dheera_ai proxy server with your config
- Sign into Dheera AI UI via SSO
- Navigate to `Teams` and you should see the new team `Default Dheera AI Prod Team` auto-created on Dheera AI
- Note Dheera AI will set the default params for this new team. 

<Image img={require('../../img/msft_default_settings.png')}  style={{ width: '900px', height: 'auto' }} />


## 4. Using Entra ID App Roles for User Permissions

You can assign user roles directly from Entra ID using App Roles. Dheera AI will automatically read the app roles from the JWT token during SSO sign-in and assign the corresponding role to the user.

### 4.1 Supported Roles

Dheera AI supports the following app roles (case-insensitive):

- `proxy_admin` - Admin over the entire Dheera AI platform
- `proxy_admin_viewer` - Read-only admin access (can view all keys and spend)
- `org_admin` - Admin over a specific organization (can create teams and users within their org)
- `internal_user` - Standard user (can create/view/delete their own keys and view their own spend)

### 4.2 Create App Roles in Entra ID

1. Navigate to your App Registration on https://portal.azure.com/
2. Go to **App roles** > **Create app role**

3. Configure the app role:
   - **Display name**: Proxy Admin (or your preferred display name)
   - **Value**: `proxy_admin` (use one of the supported role values above)
   - **Description**: Administrator access to Dheera AI proxy
   - **Allowed member types**: Users/Groups


4. Click **Apply** to save the role

### 4.3 Assign Users to App Roles

1. Navigate to **Enterprise Applications** on https://portal.azure.com/
2. Select your Dheera AI application
3. Go to **Users and groups** > **Add user/group**
4. Select the user and assign them to one of the app roles you created


### 4.4 Test the Role Assignment

1. Sign in to Dheera AI UI via SSO as a user with an assigned app role
2. Dheera AI will automatically extract the app role from the JWT token
3. The user will be assigned the corresponding Dheera AI role in the database
4. The user's permissions will reflect their assigned role

**How it works:**
- When a user signs in via Microsoft SSO, Dheera AI extracts the `roles` claim from the JWT `id_token`
- If any of the roles match a valid Dheera AI role (case-insensitive), that role is assigned to the user
- If multiple roles are present, Dheera AI uses the first valid role it finds
- This role assignment persists in the Dheera AI database and determines the user's access level

## Video Walkthrough

This walks through setting up sso auto-add for **Microsoft Entra ID**

Follow along this video for a walkthrough of how to set this up with Microsoft Entra ID

<iframe width="840" height="500" src="https://www.loom.com/embed/ea711323aa9a496d84a01fd7b2a12f54?sid=c53e238c-5bfd-4135-b8fb-b5b1a08632cf" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>













