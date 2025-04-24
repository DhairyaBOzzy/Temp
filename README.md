
# BAPS Columbus Registration Portal

<p align="center">
  <img src="/public/baps-logo.svg" alt="BAPS Columbus" height="110"/>
</p>

A **Vite + React (TypeScript)** single-page application for collecting and managing member registrations at BAPS Columbus Mandir. The form dynamically adapts to household composition, performs strong client-side validation, blocks duplicates & spam with Firestore queries, and persists everything securely in **Firebase Cloud Firestore**.

---

## ✨ Key Features

| Domain | Details |
|--------|---------|
| **Dynamic UI** | Spouse panel appears when *Marital Status = Married* · Children panel can add ≤ 4 children with table preview & row delete |
| **Validation & Formatting** | Live regex checks for email · Phone auto-formats *(###) ###-####* and enforces 10 digits · 5-digit ZIP filter |
| **Data Integrity** | Composite index driven **duplicate** checks on `primaryMember.email` & `primaryMember.phone` · **Rate-limit** = 5 min between submissions per email |
| **Resilience** | Toast notifications for every success/error · Graceful Firestore error handling |
| **Developer UX** | Absolute imports via `@/` alias · ESLint + Prettier + Husky pre-commit · Tailwind CSS with shadcn/ui and Radix primitives |
| **Deploy Anywhere** | Vite `base:""` ready for Firebase Hosting, Vercel, Netlify |

---

## 🗂 Project Structure

```
📦baps-registration
├─ .env.example              # copy -> .env for local secrets
├─ package.json              # scripts & dependencies (see below)
├─ vite.config.ts            # alias @ → src/, base="" for hosting
├─ tsconfig.json             # paths config
├─ firebase.json             # hosting + emulators
├─ firestore.rules           # *public-read, create-only* rules
├─ firestore.indexes.json    # composite index for duplicate & rate-limit queries
├─ public/
│   ├─ baps-logo.svg         # production logo
│   └─ index.html
└─ src/
    ├─ firebase.ts           # initialiseApp + getFirestore
    ├─ pages/Index.tsx       # main wizard + submit logic
    ├─ components/
    │   ├─ PrimaryMemberForm.tsx
    │   ├─ SpouseForm.tsx
    │   ├─ ChildrenForm.tsx
    │   ├─ FormSummary.tsx
    │   └─ ui/… (shadcn re-exports)
    └─ styles/tailwind.css
```

---

## 🔧 Prerequisites

| Requirement | Version |
|-------------|---------|
| **Node JS** | ≥ 18.x |
| **pnpm / npm / yarn** | latest |
| **Firebase CLI** | `npm i -g firebase-tools` |

---

## 📦 Dependencies Snapshot

> For the full list open *package.json*, but highlights include:

* **react 18**, **@vitejs/plugin-react-swc** – super-fast dev/TSX build
* **tailwindcss 3** + **tailwindcss-animate** + **clsx** + **class-variance-authority**
* **Radix UI** + **shadcn/ui** components used throughout
* **firebase 11.6** – Firestore client
* **lucide-react**, **sonner (toast)**, **zod** (future schema validation)

---

## 🛠 NPM Scripts

| Script | Purpose |
|--------|---------|
| `pnpm dev` | Launch Vite dev server on <http://localhost:8080> |
| `pnpm build` | Production build to `dist/` |
| `pnpm build:dev` | *Debug* build without minification |
| `pnpm preview` | Serve the production bundle locally |
| `pnpm lint` | ESLint + Prettier checks |

---

## 🔐 Environment Variables (`.env`)

```
VITE_FIREBASE_API_KEY="…"
VITE_FIREBASE_AUTH_DOMAIN="…"
VITE_FIREBASE_PROJECT_ID="…"
VITE_FIREBASE_STORAGE_BUCKET="…"
VITE_FIREBASE_MESSAGING_SENDER_ID="…"
VITE_FIREBASE_APP_ID="…"
```
> **Never commit** real keys – use `.env.local` in CI.

---

## 🔒 Firestore Security Rules (`firestore.rules`)

```firestore
service cloud.firestore {
  match /databases/{db}/documents {
    // Registrations collection is *append-only*
    match /registrations/{id} {
      allow create: if true;      // anyone can submit
      allow read, list: if true;  // anyone can view (adjust as needed)
      allow update, delete: if false;
    }
    match /{document=**} {
      allow read, write: if false; // deny everything else
    }
  }
}
```
> Tighten reads with auth / custom claims when ready.

---

## 🗄 Composite Index

The duplicate & recent-submission queries require:

```jsonc
{
  "collectionGroup":"registrations",
  "fields":[
    {"fieldPath":"primaryMember.email","order":"ASCENDING"},
    {"fieldPath":"submittedAt","order":"DESCENDING"}
  ]
}
```
Create automatically via the link in the Firebase error dialog, or import `firestore.indexes.json` in the console.

---

## 🧪 Form Logic & Validation

### Primary Member
* Mandatory fields: first/last name, address, city, state, zip (5), email, phone (10), gender, maritalStatus
* Email regex, phone formatter, zip filter built into **`handleChange`** in `PrimaryMemberForm.tsx`

### Spouse
* Shown only when maritalStatus = *Married*
* First & last name required; email & phone optional but validated if filled

### Children (≤ 4)
* Each child requires first/last name, gender, full DOB (day/month/year)
* Optional email per child
* Table summary + row deletion with `<Trash2/>`

### Submission Flow (`Index.tsx`)
1. **validateForm()** guards required fields.
2. Duplicate‐email query ➔ toast error.
3. Duplicate‐phone query ➔ toast error.
4. *Rate-limit* query on `submittedAt` within last 5 min.
5. If all clear: `addDoc()` to `registrations` and show `<FormSummary/>`.

---

## 🚀 Local Dev & Emulators

```bash
# serve + hot-reload
pnpm dev

# optionally spin up Firestore emulator (port 8080)
firebase emulators:start --only firestore
```
*The Vite dev server uses port 8080 to avoid clash with the emulator.* Adjust in `vite.config.ts` if you change emulator ports.

---

## 🏗 Deployment

### Firebase Hosting

```bash
firebase login
firebase init hosting  # choose dist/ as publicDir
pnpm build
firebase deploy
```

### Vercel / Netlify
* Just add the same **env vars** in project settings.
* Build command: `pnpm build`, output: `dist/` (detected automatically).

---


## 🪪 License

MIT © 2025 BAPS Columbus, OH
