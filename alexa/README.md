# The Alexa skill

A custom skill: the person says "Alexa, open knots partner", then "about memecoins" or "what should I explore", and
Knots answers with a pick and a reason. It calls `POST /alexa` on your server, which checks Amazon's signature on
every request before doing any work (`knots/alexa_skill.py`).

## Set it up

1. Deploy the server first (`docs/DEPLOY_EC2.md`) so `https://<DOMAIN>/alexa` exists.
2. Alexa developer console -> Create Skill -> **Custom**, English (US), and a "provision your own" backend.
3. Build -> **JSON Editor**: paste `skill-package/interactionModels/custom/en-US.json`. Save and build the model.
4. Build -> **Endpoint** -> HTTPS: `https://<DOMAIN>/alexa`, certificate type "My development endpoint has a
   certificate from a trusted certificate authority".
5. Copy the **Skill ID** (`amzn1.ask.skill...`) into `.env` as `KNOTS_ALEXA_SKILL_ID`, then
   `docker compose up -d`. Until it is set the route answers 503, and requests for any other skill are refused.
6. Test tab -> enable testing -> try: "open knots partner", "about memecoins", "good pick".

You can instead use the ASK CLI with `skill-package/` (replace `YOUR-DOMAIN` in `skill.json` first).

## How it behaves

- **Memory:** the person is Alexa's `userId` (from a verified request), stored only as a salted hash. No account linking.
  I believe the id can change if a person disables and re-enables the skill, which would start a fresh memory; I
  have not verified that.
- **Feedback:** after picks, the reprompt asks "Was <title> a good pick? Say good pick, or not for me." Either
  answer (or saying it any time) records a verdict for the first pick that was read out. Those verdicts are the labels
  retraining learns from.
- **The 8-second limit:** Alexa gives a skill about 8 seconds. The pipeline (Bedrock, three catalogs, embeddings) can
  take longer, so it runs under a 6.5 second deadline (`KNOTS_ALEXA_TIMEOUT_S`). If missed, the person hears "ask me
  again in a moment", the work continues, and asking the same thing within two minutes returns the finished answer.
  There are no progressive responses yet. **I have not measured real latency**, so check it in the simulator.
- **Invocation name:** `knots partner` (Alexa wants at least two words). Change it in the interaction model and
  `skill.json` if you prefer another.

## Before publishing

Certification needs a real privacy policy URL (`skill.json` has a placeholder) and a review of the invocation name.
Nothing here has been run on a device or the simulator.
