# Environment Setup — Read This Before Module M1.1

Do this once, carefully, before the first lab. It takes about 5 minutes and avoids almost every
problem that comes up during the labs. If something doesn't match what you see on screen, ask
immediately — don't guess and don't skip a step.

---

## Step 1 — Confirm you can see your files

You should have this repository already cloned onto your VM. Open it in VS Code. You should see
folders named `Day1`, `Day2`, `Day3`. If you don't see this, ask now before continuing.

## Step 2 — Get your OpenAI API key into a `.env` file

You were sent an API key by email. **Do not paste it into a Python cell. Do not paste it into any
chat, Slack, WhatsApp, or forum. It is like a password.**

1. Open your email **using the browser inside this VM**, not your personal laptop. (If you open it
   on your laptop, copy-paste from your laptop into this VM will not work — that's a limitation of
   this lab environment, not something you're doing wrong.)
2. Copy the key from the email — inside the VM, copying and pasting between two windows on the
   *same* machine works fine.
3. In VS Code, inside the `Day1/labs/starter/` folder, create a new file named exactly `.env`
   (yes, starting with a dot, no other extension — if your file ends up named `.env.txt`, rename it
   to remove the `.txt`).
4. Put one line in it, with your real key:
   ```
   OPENAI_API_KEY=sk-...your-key-here...
   ```
   No quotes, no spaces around the `=`. Save the file.

**One file for all three days.** You create this `.env` once, in `Day1/labs/starter/`. Every
notebook on Day 2 and Day 3 searches for it automatically — you never need to copy it into another
folder. If you prefer, you can instead put it at the top level of the repo (next to `SETUP.md`) and
that works too.

---

## Step 3 — Install one extra package

Open a terminal in VS Code (View → Terminal) and run:
```
pip install python-dotenv
```
If it says "Requirement already satisfied," that's fine — it means it's already there.

## Step 4 — Open the notebook and select a kernel

1. Open `Day1/labs/starter/D1_M1.1_LLM_Mechanics_Starter.ipynb`.
2. If VS Code asks you to "Select Kernel," choose **Python Environments...**, then pick the Python
   version shown (don't worry about which one unless there are several — ask if unsure).
3. The very first time you do this, VS Code may show a spinner for a minute or two while the kernel
   starts up. That's normal, not a hang — wait for it.

## Step 5 — Load your key into the notebook

At the very top of the notebook, **before** the "Setup" cell, add a new code cell (hover just below
any cell and click **+ Code**, or use the **+ Code** button in the toolbar) with exactly this,
adjusting the path if your folder location is different:

```python
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Administrator\EY_GenAI_3D\Day1\labs\starter\.env", override=True)
```

Run this cell first (click the ▷ next to it).

## Step 6 — Verify it worked

Add another new cell right after it:
```python
import os
print("Key is set:", os.environ.get("OPENAI_API_KEY") is not None)
```
Run it. You must see `Key is set: True` before continuing. If you see `False`:
- Check the `.env` file is really named `.env`, not `.env.txt`
- Check the path in Step 5 matches exactly where your `.env` file actually is
- Check the file has no extra blank lines or typos

## Step 7 — Run the Setup cell

Now run the notebook's actual "Setup" cell (the one starting with `import os`). It should complete
in a couple of seconds with no error. If you get an `AssertionError`, go back to Step 6 — your key
isn't being found yet.

## Step 8 — Now you're ready for Task 1

Read the explanation above each task before running its code cell. Run cells **in order, top to
bottom** — skipping around or running Task 3 before Task 1 will cause confusing errors that have
nothing to do with your actual work.

---

## The one rule that fixes almost everything

**If you change your `.env` file, or if anything seems stuck or wrong after an error: click
Restart in the toolbar, then run every cell again from the top, in order.** A restart clears
everything Python has remembered, including your old key — editing a file on disk does not
automatically update a Python session that's already running. This single step solves the large
majority of confusing errors in this lab.

## If you get an `AuthenticationError` (401, "incorrect API key")

This means the key itself is wrong — usually an incomplete copy-paste, not a code problem.
1. Open your `.env` file and visually check the value starts with `sk-`.
2. If it doesn't, go back to your email, re-copy the full key carefully, replace the line in
   `.env`, save.
3. **Restart the kernel** (see above — this is required, not optional).
4. Run cells from the top again.

## What never to do

- Never paste your API key into a code cell directly — always through `.env`.
- Never paste your API key into chat, WhatsApp, or any message to anyone, including the trainer.
- Never commit or push a `.env` file to GitHub — it's already excluded for you, don't remove that
  protection.
