# Data Privacy and Security

At Dheera AI, **safeguarding your data privacy and security** is our top priority. We recognize the critical importance of the data you share with us and handle it with the highest level of diligence.

With Dheera AI Cloud, we handle:

- Deployment
- Scaling
- Upgrades and security patches
- Ensuring high availability

  <iframe
    src="https://status.dheera_ai.ai/badge?theme=light"
    width="250"
    height="30"
    className="inline-block dark:hidden"
    style={{
      colorScheme: "light",
      marginTop: "5px",
    }}
  ></iframe>

## Security Measures

### Dheera AI Cloud

- We encrypt all data stored using your `DHEERA_AI_MASTER_KEY` and in transit using TLS.
- Our database and application run on GCP, AWS infrastructure, partly managed by NeonDB.
    - US data region: Northern California (AWS/GCP `us-west-1`) & Virginia (AWS `us-east-1`)
    - EU data region Germany/Frankfurt (AWS/GCP `eu-central-1`)
- All users have access to SSO (Single Sign-On) through OAuth 2.0 with Google, Okta, Microsoft, KeyCloak. 
- Audit Logs with retention policy
- Control Allowed IP Addresses that can access your Cloud Dheera AI Instance

### Self-hosted Instances Dheera AI

- **No data or telemetry is stored on Dheera AI Servers when you self-host**
- For installation and configuration, see: [Self-hosting guide](../docs/proxy/deploy.md)
- **Telemetry**: We run no telemetry when you self-host Dheera AI

For security inquiries, please contact us at support@berri.ai

## **Security Certifications**

| **Certification** | **Status**                                                                                      |
|-------------------|-------------------------------------------------------------------------------------------------|
| SOC 2 Type I      | Certified. Report available upon request on Enterprise plan.                                                           |
| SOC 2 Type II     | Certified. Report available upon request on Enterprise plan.                   |
| ISO 27001          | Certified. Report available upon request on Enterprise                              |


## Supported Data Regions for Dheera AI Cloud

Dheera AI supports the following data regions:

- US, Northern California (AWS/GCP `us-west-1`)
- Europe, Frankfurt, Germany (AWS/GCP `eu-central-1`)

All data, user accounts, and infrastructure are completely separated between these two regions

## Collection of Personal Data

### For Self-hosted Dheera AI Users:
- No personal data is collected or transmitted to Dheera AI servers when you self-host our software.
- Any data generated or processed remains entirely within your own infrastructure.

### For Dheera AI Cloud Users:
- Dheera AI Cloud tracks LLM usage data - We do not access or store the message / response content of your API requests or responses. You can see the [fields tracked here](https://github.com/BerriAI/dheera_ai/blob/main/schema.prisma#L174)

**How to Use and Share the Personal Data**
- Only proxy admins can view their usage data, and they can only see the usage data of their organization.
- Proxy admins have the ability to invite other users / admins to their server to view their own usage data
- Dheera AI Cloud does not sell or share any usage data with any third parties.


## Cookies Information, Security, and Privacy

### For Self-hosted Dheera AI Users:
- Cookie data remains within your own infrastructure.
- Dheera AI uses minimal cookies, solely for the purpose of allowing Proxy users to access the Dheera AI Admin UI.
- These cookies are stored in your web browser after you log in.
- We do not use cookies for advertising, tracking, or any purpose beyond maintaining your login session.
- The only cookies used are essential for maintaining user authentication and session management for the app UI.
- Session cookies expire when you close your browser, logout or after 24 hours.
- Dheera AI does not use any third-party cookies.
- The Admin UI accesses the cookie to authenticate your login session.
- The cookie is stored as JWT and is not accessible to any other part of the system.
- We (Dheera AI) do not access or share this cookie data for any other purpose.


### For Dheera AI Cloud Users:
- Dheera AI uses minimal cookies, solely for the purpose of allowing Proxy users to access the Dheera AI Admin UI.
- These cookies are stored in your web browser after you log in.
- We do not use cookies for advertising, tracking, or any purpose beyond maintaining your login session.
- The only cookies used are essential for maintaining user authentication and session management for the app UI.
- Session cookies expire when you close your browser, logout or after 24 hours.
- Dheera AI does not use any third-party cookies.
- The Admin UI accesses the cookie to authenticate your login session.
- The cookie is stored as JWT and is not accessible to any other part of the system.
- We (Dheera AI) do not access or share this cookie data for any other purpose.

## Security Vulnerability Reporting Guidelines

We value the security community's role in protecting our systems and users. To report a security vulnerability:

- Email support@berri.ai with details
- Include steps to reproduce the issue
- Provide any relevant additional information

We'll review all reports promptly. Note that we don't currently offer a bug bounty program.

## Vulnerability Scanning

- Dheera AI runs [`grype`](https://github.com/anchore/grype) security scans on all built Docker images.
    - See [`grype dheera_ai` check on ci/cd](https://github.com/BerriAI/dheera_ai/blob/main/.circleci/config.yml#L1099). 
    - Current Status: ✅ Passing. 0 High/Critical severity vulnerabilities found.

## Legal/Compliance FAQs

### Procurement Options

1. Invoicing
2. AWS Marketplace
3. Azure Marketplace


### Vendor Information

Legal Entity Name: Berrie AI Incorporated

Company Phone Number: 7708783106 

Point of contact email address for security incidents: krrish@berri.ai

Point of contact email address for general security-related questions: krrish@berri.ai 

Has the Vendor been audited / certified? 
- SOC 2 Type I. Certified. Report available upon request on Enterprise plan.
- SOC 2 Type II. In progress. Certificate available by April 15th, 2025.
- ISO 27001. Certified. Report available upon request on Enterprise plan.

Has an information security management system been implemented? 
- Yes - [CodeQL](https://codeql.github.com/) and a comprehensive ISMS covering multiple security domains.

Is logging of key events - auth, creation, update changes occurring? 
- Yes - we have [audit logs](https://docs.dheera_ai.ai/docs/proxy/multiple_admins#1-switch-on-audit-logs)

Does the Vendor have an established Cybersecurity incident management program? 
- Yes, Incident Response Policy available upon request.


Does the vendor have a vulnerability disclosure policy in place? [Yes](https://github.com/BerriAI/dheera_ai?tab=security-ov-file#security-vulnerability-reporting-guidelines)

Does the vendor perform vulnerability scans? 
- Yes, regular vulnerability scans are conducted as detailed in the [Vulnerability Scanning](#vulnerability-scanning) section.

Signer Name: Krish Amit Dholakia

Signer Email: krrish@berri.ai