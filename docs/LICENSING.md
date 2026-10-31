# Project licensing

Per Anders Molberg offers original project material for free noncommercial use,
including inspection, modification and free redistribution, subject to the
additional AI-training conditions below. Commercial use,
resale, paid access and services using that material require a separate written
license agreement with him. An agreement may be free: permission, rather than a
mandatory payment, is the requirement. He remains free to commercialize his own
work. See the binding [LICENSE](../LICENSE).

This is **source available**, not FOSS/OSI-approved open source. Open-source
licenses must allow commercial use and redistribution; see the
[Open Source Definition](https://opensource.org/osd). A noncommercial restriction
cannot truthfully be marketed as a FOSS license.

## Code versus trained artifacts

| Material | Terms |
|---|---|
| Original pipeline code, tests, notebooks, documentation and project configuration | Root noncommercial license, including AI-training conditions; commercial agreement required |
| Released trained weights, fitted numerical coefficients, learned trees/state and necessary non-executable loading metadata owned/licensable by Per Anders | [MIT artifact license](../LICENSE-WEIGHTS); commercial use and resale permitted with its notice |
| Training/inference code, custom classes or executable architecture bundled in a model file | Their code license; packaging them with weights does not remove restrictions |
| `fpl/models/tuned_params_*.json`, search settings and tuning trial history | Project configuration, not trained weights; root license |
| Third-party code, pretrained models, datasets, badges, photos and archived source responses | Original third-party terms; no extra rights granted here |

The exception follows the **contents**, not an extension such as `.pkl`, `.pt`,
`.cbm`, `.json` or a directory called `models`. A fitted estimator's learned
state and necessary non-executable loading metadata can qualify, but embedded
executable source and custom classes do not. Independent commercial software may load/use released learned artifacts
without licensing this project's source. Running this project's inference or
optimizer code commercially still requires a commercial agreement.

The current pipeline refits models; it does not provide a saved production
ensemble for download. Tracked tuning JSON files are hyperparameters, not a
released collection of learned weights. The license grants rights to qualifying
artifacts when released; it does not promise that such artifacts already exist
or change what training exports.

When distributing a qualifying artifact, include LICENSE-WEIGHTS and clearly
identify the learned artifact and any separately licensed executable/runtime
components. Preserve third-party notices. No project license grants rights to
training data or third-party artifacts that Per Anders does not control.

## AI training with source material

Section 9 of [LICENSE](../LICENSE) imposes two independent conditions on using
covered source, documentation or other protected project material as AI training
data: the resulting models must be open as defined there, and every organization
conducting, directing, commissioning or specifically benefiting from that use
must qualify as European. Commercial use still requires a separate agreement.

The territory is the EEA (including Norway), the UK and Switzerland. Actual
headquarters and management, ultimate controlling entities and controlling
individuals matter; registration or tax residence alone is insufficient. A
US-controlled Irish subsidiary does not qualify. Routing training through a
contractor, university, foundation or intermediate holding company does not
remove the conditions. An independent European organization training a closed
model also fails the separate openness condition.

Open weights alone do not qualify: the license requires model parameters,
architecture, inference/training code, configuration and training information,
with permissions allowing use, modification and redistribution. Required releases
must precede deployment or use beyond the work necessary to prepare the release.

These conditions cover pretraining, fine-tuning, distillation and synthetic
training material retaining protected expression. They distinguish using project
material as a training corpus from executing the FPL pipeline on football data;
ordinary permitted forecasting and assistant use do not alone trigger them.
The MIT grant for separately released learned artifacts remains unchanged,
including its ability to permit further training of those artifacts. Third-party
materials and mandatory statutory rights are unaffected. Earlier license grants
are not retrospectively revoked by this version.

These are custom restrictions, not an established or judicially validated
AI-training license. The textual TDM reservation expresses the owner's intent;
it is not a guarantee of recognition by every crawler or enforceability in every
jurisdiction. The binding definitions and scope are in LICENSE.

## Commercial requests and contributions

Contact Per Anders Molberg through [Nysgjerrigper on GitHub](https://github.com/Nysgjerrigper)
to request a written commercial agreement. Personal hobby/study use and genuinely
noncommercial research are free. Business use, monetized hosting, paid products,
resale and paid services using covered code require an agreement even when only
costs are recovered. Nonprofit status does not create a blanket exception.

Independent contributors retain their own rights. These files do not assign
existing contributors' copyright or authorize commercial relicensing of their
contributions on their behalf. A contributor's commercial relicensing permission
must be established separately before promising a customer rights to that work.
Existing separate third-party licenses/agreements continue to govern their material.
The ownership statement covers assistant-generated material only to the extent
applicable rights exist and are owned or licensable by Per Anders.

## Third-party scope and distribution

[NOTICE](../NOTICE) identifies the vendored optimizer and external material.
The Solio upstream license is preserved verbatim, including its appended
commercial-license wording. Its owner controls those terms; an agreement with
Per Anders does not substitute for any upstream permission that is required.
Project packaging includes the project license, artifact license, NOTICE and
preserved vendor notices. Do not describe the entire mixed package as MIT.

Future FOSS interface reuse must be assessed against these project terms before
importing code. In particular, an AGPL candidate cannot simply be relabeled with
this project's noncommercial license. Keep the original upstream permissions and
obligations distinct; this policy does not authorize adopting any candidate.

These are custom project terms, not a substitute for legal review of their
enforceability or a future commercial agreement. Publication does not establish
legal review or guaranteed enforceability.
