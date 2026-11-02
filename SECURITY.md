# Security policy

## Report a vulnerability

Email **yanchaomei@mail.ustc.edu.cn** with the subject “InferenceForge security report”. Include the affected commit/version, impact, and a minimal reproduction with synthetic inputs. Use this private contact for vulnerabilities instead of a public issue.

Do not include live credentials, private model-service keys, or business prompts. The maintainer will coordinate reproduction, a fix, and disclosure by email. Reports are reviewed against the current source and latest release; older releases do not have a separate maintained security branch.

## Running the service

The HTTP service listens on `127.0.0.1` by default and has no built-in authentication or TLS. The SDK example's `api_key="unused"` provides no protection. Use SSH forwarding for personal remote access, or your existing authenticated TLS gateway when exposing another interface.

Construction runs generated Python code. Use a dedicated execution environment and only the credentials and data needed for the task. Keep generator credentials on the controller, out of model artifacts and public logs.
