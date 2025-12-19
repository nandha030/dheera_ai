Additional files for the proxy. Reduces the size of the main dheera_ai package.

Currently, only stores the migration.sql files for dheera_ai-proxy.

To install, run:

```bash
pip install dheera_ai-proxy-extras
```
OR 

```bash
pip install dheera_ai[proxy] # installs dheera_ai-proxy-extras and other proxy dependencies
```

To use the migrations, run:

```bash
dheera_ai --use_prisma_migrate
```

