# Friction Log

Project: Knots (MCP server for Alexa+). Each entry records the task, steps,
expected and actual results, severity, workaround, and a suggestion.

## 1. Alexa+ add-on blocked by AWS role-assumption errors

- **Task attempted:** Register Knots as an Alexa+ add-on so Alexa+ can call the MCP server.
- **Steps taken:** Deployed the MCP server behind Caddy with HTTPS, wrote the add-on definition and tool descriptions, then tried to register the add-on and set up the AWS role it needs.
- **Expected:** The role is assumed, the add-on registers, and Alexa+ calls the server.
- **Actual:** Role-assumption (IAM) errors blocked setup, so the add-on was never registered. **Exact error message or code:** not yet recorded.
- **Severity:** High. It blocked the live Alexa+ demo.
- **Workaround:** Built the Explore web UI to act as the MCP client. It calls the same tools and shows the same picks, notes, and spoken line Alexa+ would receive.
- **Suggestion:** Make the error name the role, the trust policy, and the missing permission, and provide a worked example of the role setup for MCP add-ons. A local MCP test client that simulates Alexa+ calls would let developers test before registration works.

## 2. YouTube API quota exhausted during live testing

- **Task attempted:** Fetch explainer videos for each interest during live test sessions.
- **Steps taken:** Ran repeated recommendation sessions against the real YouTube Data API.
- **Expected:** Explainer results on every request.
- **Actual:** Search stopped returning results once the daily quota was used up, so explainer picks disappeared.
- **Severity:** Medium. Recommendations still worked, but one format dropped out.
- **Workaround:** The Explorer's YouTube toggle starts off; the README notes that YouTube allows about 100 searches a day for the whole server. The specific workaround used during the live sessions is not recorded.
- **Suggestion:** Show remaining quota more clearly and return an explicit "quota exceeded" error. Search calls cost far more quota than ordinary reads, and that cost should be easier to see up front.

## 3. Proactive recommendations are hard with MCP

- **Task attempted:** Have Knots surface recommendations without being asked.
- **Steps taken:** Looked for a way for the server to push suggestions to Alexa+.
- **Expected:** A way to notify the user or trigger a call on a schedule.
- **Actual:** An MCP server only responds when called, so it can't initiate anything.
- **Severity:** Medium. It shapes the product, since Knots only answers when asked.
- **Workaround:** Designed the main entry point around the prompt "what are we doing today?", so recommendations arrive when Alexa+ asks.
- **Suggestion:** Support scheduled or event-triggered tool calls for Alexa+ add-ons, or a push/notification channel for MCP servers.
