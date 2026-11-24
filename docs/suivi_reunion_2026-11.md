# Suivi des objectifs de la réunion : simulation CaT-MPPI avec un humain

Mathilde Lalanne, Gepetto (LAAS-CNRS). Version du 24 novembre 2026, avec les résultats sur 30 runs par config (§ 4.4).

Objectifs fixés en réunion :

1. rapprocher l'humain pour le mouvement vertical du bras ;
2. tester un mouvement de la main qui vient bloquer B ; si B est bloqué, le robot attend ;
3. sur ce même mouvement, ajouter des mouvements rapides aléatoires.

**Résumé**

| Objectif | Statut | Résultat principal |
|---|---|---|
| 1. Humain plus proche | fait | x −1,10 → x −1,00 : le bras gêne vraiment le robot. Aucun réglage ne supprime tous les contacts à 0,5 Hz (2 à 5 runs sur 10 avec contact). |
| 2. Bras qui bloque B | fait | Sur 342 gestes, le robot n'arrive jamais à B tant que la main y est. Il attend à environ 16 cm, sans aucune règle au niveau tâche : c'est MPPI seul qui décide. |
| 3. Mouvements rapides aléatoires | fait (deux scènes, 30 runs par config) | Bras qui bloque B, avec à-coups et `clip10` : 0 contact sur 30 runs, au moins 2,5 cm du bras ; 0 arrivée à B main sur B sur 678 gestes. Bras au milieu : les à-coups augmentent nettement les contacts (`cfix12` : 3 → 9/30) ; 43 % des contacts commencent pendant un à-coup. |

---

## 1. Cadre commun

- **Simulation :** Franka FER, Gazebo + ROS 2 Humble. Pick-and-place en boucle entre A (−0,5 ; −0,5 ; 0,3) et B (−0,5 ; 0,5 ; 0,3), dans le repère `fer_link0`.
- **Contrôleur :** CaT-MPPI (STORM), mode standard uniquement.
  - Le contournement, l'attente et le choix dessus/dessous viennent de MPPI.
  - Aucune règle au niveau tâche.
- **Contrainte humaine :** marge de 10 cm autour des capsules des bras humains, jamais réduite.
  - Le mur et la table sont aussi des contraintes.
  - Pour les contacts, seuls les bras humains comptent ; le corps est ignoré.
- **Réglage de base de toutes les campagnes :** `cattab + b75 + pmax15`, issu des campagnes du 16–17 novembre.
  - `cattab` : la table est traitée comme une contrainte CaT (p_max 0,15) et non plus comme un coût de poids 1000. Ce coût gonflait la référence de coût de CaT et faisait caler le robot.
  - `b75` : beta MPPI 75 au lieu de 25.
  - `pmax15` : p_max de la contrainte humaine à 0,15 au lieu de 0,05.
- **Prévision du bras :** analytique, publiée comme capsules futures.
  - L'horizon s'adapte à la vitesse observée du bras, de 1 s (bras lent) à 2 s (bras rapide).
  - Elle ne connaît pas l'avenir aléatoire : ni les changements d'amplitude, ni le départ des gestes, ni les à-coups.
- **Protocole :** chaque run dure 230 s simulées (environ 4,25 min réelles). En général 5 fréquences de bras (0,1 à 0,5 Hz) × 2 runs par config.
  - La graine humaine vaut rep × 100 + 10 f. Elle est identique d'une config à l'autre, donc les comparaisons sont appariées.
- **Critère de classement :** d'abord aucun contact avec un bras humain ; ensuite le nombre d'arrivées à B.
- **Mesures :**
  - contact = pénétration d'une capsule de bras ;
  - distance minimale robot–bras ;
  - arrivées à A et B ;
  - abandons (cible non atteinte en 110 s).
- **Validité :** un run où deux simulations ont tourné en même temps (plus de 32 messages joint_states par seconde simulée) est écarté. Aucun run de ce document n'est concerné.

## 2. Objectif 1 : humain plus proche, mouvement vertical

**Situation de départ.** Humain à x −1,10, épaule droite centrée à 1,2 rad, sinus vertical dont l'amplitude alterne au hasard entre 0,15 et 0,45 rad à chaque période. À cette distance, le bras gênait peu le robot : 14,4 arrivées à B par run.

**Rapprochement** (campagnes repeat-wkX2 et repeat-wkX3, 20 novembre)

| Position | Config | Runs avec contact | Arrivées à B par run |
|---|---|---|---|
| x −1,00 | référence | 5/10 | 6,0 |
| x −0,95 | référence | 8/10 | n.c. |
| x −1,00 | `clip10` | 3/10 | n.c. |
| x −1,00 | `pmax20` | 3/10 | n.c. |
| x −1,00 | `cfix12` | 3/10 | n.c. |
| x −1,00 | `core3` (zone dure, pour comparaison) | 2/10 (jusqu'à −10,7 cm) | n.c. |

*n.c. : non relevé dans ce document.*

- x −1,00 a été retenu : le bras gêne vraiment le robot. 31 des 38 runs avec contact touchent en transit, sous l'avant-bras droit, qui couvre tout le passage A→B à cette distance.
- À x −0,85, il y avait un contact dans 168 runs sur 168 : la posture du robot à B entre d'environ 12,7 cm dans la zone balayée par le bras.

**Bras au milieu du trajet** (flag `mid`, épaule centrée à 0,9 rad) : l'avant-bras balaie exactement le milieu du trajet A→B. Campagnes repeat-wkM1 et repeat-wkM2, 23 novembre, sans zone dure :

| Config | Runs avec contact | Fréquences touchées | Pire contact | Distance min. sans contact (médiane) | Arrivées à B par run | Abandons |
|---|---|---|---|---|---|---|
| référence `pmax15` | 5/10 | 0,2 · 0,4 · 0,4 · 0,5 · 0,5 | −7,2 cm | 0,9 cm | 10,5 | 1 |
| `clip10` (écrêtage de la violation 0,1 au lieu de 0,2) | 3/10 | 0,4 · 0,5 · 0,5 | −3,8 cm | 2,9 cm | 9,9 | 1 |
| `pmax20` (p_max 0,20) | 5/10 | 0,1 · 0,2 · 0,4 · 0,5 · 0,5 | −12,2 cm | 2,7 cm | 10,3 | 1 |
| `cfix12` (normalisation fixe des contraintes, 1,2) | 2/10 | 0,5 · 0,5 | −7,2 cm | 6,6 cm | 9,1 | 2 |

- **Tous les contacts ont lieu en transit**, pince au milieu du trajet, sous l'avant-bras droit.
- **`cfix12`** garde le plus de distance de 0,1 à 0,4 Hz, sans aucun contact. À 0,5 Hz, il se bloque sous le bras : contact dans les 2 runs, 4 arrivées à B, un abandon par run.
- **`pmax20`** passe trop bas sous le bras quand celui-ci descend.
- **À 0,5 Hz, aucun réglage n'atteint zéro contact.**

## 3. Objectif 2 : un bras qui bloque B, et le robot attend

Trois versions se sont succédé avant la version finale.

| Version | Ce qui a été fait | Pourquoi elle a été abandonnée |
|---|---|---|
| a. Main gauche périodique au-dessus de B + règle d'arrêt au niveau tâche (`goal_block_distance`) | Le bras gauche avance sur B toutes les 12 s (1 s d'avance, 4 s sur B, 1 s de recul). Le gestionnaire de tâches peut geler la cible quand B est bloqué. | La campagne repeat-wkB1 a montré que **MPPI seul attend déjà** (pince jamais à moins de 9,8 cm de B pendant le blocage) : la règle d'arrêt n'apporte rien et contredit le principe « MPPI décide ». Il fallait de plus un mouvement d'avant en arrière du bras droit. |
| b. Bras droit périodique | Humain décalé en y 0,70 pour que l'épaule droite fasse face à B. Le bras droit avance sur B (épaule Z + coude). | Le cycle du robot (environ 12,8 s) s'est calé sur celui de la main (12 s) : le bras bloquait B pendant que le robot était à A. |
| **c. Bras droit réactif (version finale)** | Le geste démarre quand la pince approche de B. Le robot ne reçoit aucune commande ; seul le bras humain réagit. | n/a |

**Version finale** (flags `y70 + blk + blkR + react`, nœud `urdf_human`, `block_mode: reactive`)

- **Déclenchement :** la distance pince–B diminue et se trouve entre `block_trigger_near` (0,20 m) et `block_trigger_far`. Période réfractaire de 3 s après chaque geste. Lecture par TF de `fer_hand_tcp`.
- **Mouvement :** articulations épaule Z et coude droits, de [0 ; 1,725] (bras replié) à [0,80 ; 0,65] rad (main sur B, à 6 mm près).
- **Avance et recul :** demi-sinusoïde de durée 1/(2f), où f est la fréquence de la campagne. La main reste 4 s sur B entre les deux.
- **Réglage selon la fréquence :** plus le geste est lent, plus il part tôt, pour que la main arrive sur B à peu près en même temps que le robot.

| f | Avance / recul | Le geste part quand la pince est à |
|---|---|---|
| 0,1 Hz | 5 s | 95 cm de B |
| 0,2 Hz | 2,5 s | 78 cm |
| 0,3 Hz | 1,7 s | 57 cm |
| 0,4 Hz | 1,25 s | 46 cm |
| 0,5 Hz | 1 s | 40 cm |

**Résultats** (campagnes repeat-wkB5 et repeat-wkB6, 23 novembre, sans zone dure, avec les à-coups du § 4)

| Config | Runs avec contact | Au plus près | Arrivées à B par run | Arrivées à B main sur B | Pince → B, main sur B (médiane) | Délai jusqu'à B après le recul (médiane) |
|---|---|---|---|---|---|---|
| référence `pmax15` | 2/10 (0,5 Hz, les deux) | −1,2 cm | 11,0 | 0/114 | 15,8 cm | 2,0 s |
| **`clip10`** | **0/10** | **4,0 cm** | 11,2 | 0/114 | 16,5 cm | 2,0 s |
| `pmax20` | 0/10 | 0,1 cm | 11,0 | 0/114 | 16,5 cm | 2,1 s |

- **Le robot attend tant que la main est sur B,** à environ 16 cm, presque immobile (2 à 4 cm/s). Il se gare côté base, à l'opposé de l'humain.
- **Il attend, il ne contourne pas :** la pince ne cherche pas à passer par-dessus ou par-dessous la main.
- **Il repart dès que la main se retire.**
- **Les deux contacts de la référence ont lieu à 0,5 Hz, pendant que la main avance :** le robot était déjà à 6–9 cm de B et c'est la main qui arrive sur lui. `clip10` supprime ces contacts.
- À 0,1 Hz, la main reste 14 s par geste : le robot fait donc 9 arrivées à B par run au lieu de 11 à 12.

## 4. Objectif 3 : mouvements rapides aléatoires

### 4.1 Mise en œuvre (flag `jrk`)

- Des à-coups s'ajoutent sur l'épaule droite (`right_shoulder_Z`), par-dessus le geste vers B ou par-dessus le sinus.
  - Amplitude signée, tirée uniformément dans ±[0,5 ; 1] × 0,35 rad, soit 0,17 à 0,35 rad.
  - Forme a·sin²(π t / 0,5 s) : un aller-retour en 0,5 s, avec une vitesse de pointe jusqu'à environ 2,2 rad/s.
  - Départs selon un processus de Poisson, en moyenne toutes les 6 s.
  - Graine fixée par run, donc reproductible et appariée entre configs.
- La prévision ne connaît pas les à-coups futurs. Elle garde le décalage observé à l'instant présent pendant tout l'horizon (paramètre `forecast_knows_jerks: false`).
- Pendant un à-coup, l'étiquette RViz affiche « _rapide ».

### 4.2 Sur le mouvement qui bloque B

Les campagnes repeat-wkB5 et repeat-wkB6 du § 3 incluent toutes les à-coups. Avec `clip10`, il n'y a aucun contact sur les 10 runs, et le robot reste toujours à plus de 4 cm du bras.

### 4.3 Sur le mouvement vertical

Les campagnes du § 2 n'avaient **pas** d'à-coups : le seul aléa y était l'alternance d'amplitude du sinus. La campagne repeat-wkM3 refait la scène du bras au milieu (`x100+alt+cattab+b75+mid`) en ajoutant `jrk`.

- **Configs :** référence `pmax15` (midj), `clip10` (midj_clip10) et `cfix12` (midj_cfix12).
- **Taille :** 5 fréquences × 2 runs, soit 30 runs, tous valides (30,2 joint_states par seconde simulée). Campagne terminée le 23 novembre à 20 h 23.
- **Appariement :** la graine du sinus et des à-coups vaut run × 100 + 10 f. Chaque run est donc comparé au run de repeat-wkM1 (référence) ou de repeat-wkM2 (`clip10`, `cfix12`) qui a exactement le même mouvement de base, sans les à-coups.
- **Attribution des contacts :** les instants des à-coups (~39 par run) ont été recalculés depuis `urdf_human` avec la même graine, en temps simulé absolu comme la trace. Un épisode de contact est attribué à un à-coup s'il commence moins de 0,5 s après le début de celui-ci. Les à-coups occupent ~8 % du temps.

| Config | À-coups | Runs avec contact | Pire contact | Échantillons de contact | Distance min. sans contact (méd.) | B par run | Abandons |
|---|---|---|---|---|---|---|---|
| référence | non | 5/10 | −7,2 cm | 90 | 0,9 cm | 10,5 | 1 |
| référence | oui | 8/10 | −9,6 cm | 99 | 1,3 cm | 10,7 | 0 |
| `clip10` | non | 3/10 | −3,8 cm | 16 | 2,9 cm | 9,9 | 1 |
| `clip10` | oui | 6/10 | −10,3 cm | 39 | 2,8 cm | 9,3 | 2 |
| `cfix12` | non | 2/10 | −7,2 cm | 62 | 6,6 cm | 9,1 | 2 |
| **`cfix12`** | **oui** | **4/10** | −11,8 cm | 107 | 3,6 cm | 10,3 | 0 |

Distance minimale au bras par fréquence, avec à-coups (cm ; run 1 / run 2 ; négatif = contact) :

| Fréquence | référence | `clip10` | `cfix12` |
|---|---|---|---|
| 0,1 Hz | −0,3 / −1,9 | 2,5 / 3,0 | 4,6 / 0,6 |
| 0,2 Hz | −2,6 / 1,6 | 6,6 (abandon) / −0,3 | −0,1 / 4,7 |
| 0,3 Hz | −7,1 / −9,6 | 0,4 / −0,9 | 3,3 / −0,5 |
| 0,4 Hz | −0,4 / 1,0 | −0,8 / −0,3 | 4,0 / −11,8 |
| 0,5 Hz | −6,8 / −7,5 | −10,3 (abandon) / −2,9 | −2,1 / 1,7 |

Ce qu'on observe :

- **Les à-coups doublent à peu près le nombre de runs avec contact**, pour les trois configs : 5 → 8, 3 → 6 et 2 → 4 sur 10. Le classement ne change pas : `cfix12` reste devant, puis `clip10`.
- **La moitié des contacts viennent directement d'un à-coup.**
  - 17 épisodes sur 34 commencent 0,10 à 0,33 s après le début d'un à-coup, c'est-à-dire quand l'épaule est au maximum de sa déviation.
  - Ces contacts sont courts (au plus 0,25 s).
  - Contrôle : dans les runs sans à-coups, avec les mêmes instants pris comme repère, on n'en trouve qu'un sur 19, ce qui correspond au hasard.
- **La profondeur de ces contacts dépend de la config :** −0,3 à −1,6 cm avec `clip10` ; jusqu'à −7,5 cm avec la référence ; jusqu'à −10,3 cm avec `cfix12` (un seul cas).
- **En retirant les contacts dus aux à-coups**, il reste 7, 4 et 1 runs sur 10 avec contact (contre 5, 3 et 2 sans à-coups). Ces autres contacts ressemblent à ceux du § 2 : en transit, la pince sous l'avant-bras droit (y entre −0,2 et 0,3).
- **Les arrivées à B ne baissent pas.** Avec `cfix12`, elles augmentent même (9,1 → 10,3). À 0,5 Hz run 2, le blocage sous le bras vu sans à-coups (4 B, −7,2 cm) n'apparaît plus : 8 B, au plus près 1,7 cm.
- **Deux runs atypiques, du même type que des cas déjà connus :**
  - `cfix12` 0,4 Hz run 2 : 100 échantillons de contact, jusqu'à −11,8 cm. De t ≈ 53 à 57 s, q2 (−1,76 rad) et q4 (−0,07 rad) sont en butée ; le robot ne peut pas s'écarter et l'avant-bras le balaie.
  - `clip10` 0,2 Hz run 1 : une seule arrivée à B. À partir de t ≈ 60 s, la pince reste à ~9 cm de B, avec q3 à −2,82 rad (butée à −2,90). C'est le blocage de posture près de B, sans lien avec l'humain, qui reste à 12–13 cm.

Rapport avec vidéos : https://claude.ai/artifact/HYLHgnazsdpZngt7a6Wu2q

### 4.4 Sur plus de runs : 30 par config (nuit du 23 au 24 novembre)

Les campagnes repeat-wkM4 (bras au milieu, 6 configs) et repeat-wkB7 (bras qui bloque B, avec à-coups seulement, référence et `clip10` ; `pmax20` non refait) ajoutent les runs 3 à 6 de chaque fréquence. Les runs 1–2 sont ceux des campagnes précédentes, reliés par lien symbolique. Au total, 160 nouveaux runs, tous valides. Un seul a été relancé (référence bras qui bloque B, 0,5 Hz, run 6), car le contrôleur du robot ne s'était pas chargé (`load_controller` en échec) et le robot n'avait pas bougé. Le mouvement de base dépend du numéro de run et de la fréquence : la comparaison avec et sans à-coups reste appariée.

**Bras qui bloque B, avec à-coups :**

| Config | Runs avec contact | Au plus près | B par run | Abandons | Arrivées à B main sur B | Pince → B main sur B (méd.) | Délai après le recul (méd.) |
|---|---|---|---|---|---|---|---|
| référence | 3/30 (toutes à 0,5 Hz) | −1,2 cm | 11,1 | 0 | 0/341 | 15,8 cm | 2,1 s |
| **`clip10`** | **0/30** | 2,5 cm | 10,9 | 1 | 0/337 | 16,5 cm | 2,0 s |

**Bras au milieu :**

| Config | Runs avec contact | Contacts > 3 cm | Pire contact | B par run | Abandons | Épisodes pendant un à-coup |
|---|---|---|---|---|---|---|
| référence | 17/30 | 7 | −9,9 cm | 10,8 | 1 | |
| référence + à-coups | 26/30 | 12 | −11,1 cm | 10,7 | 1 | 23/61 |
| `clip10` | 9/30 | 5 | −11,6 cm | 9,9 | 3 | |
| `clip10` + à-coups | 20/30 | 9 | −11,0 cm | 10,1 | 3 | 16/37 |
| `cfix12` | 3/30 | 2 | −7,6 cm | 9,0 | 6 | |
| **`cfix12` + à-coups** | **9/30** | 3 | −11,8 cm | 9,2 | 4 | 9/13 |

Runs avec contact par fréquence (sur 6), de 0,1 à 0,5 Hz :

| Config | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 |
|---|---|---|---|---|---|
| bras qui bloque B, référence | 0 | 0 | 0 | 0 | 3 |
| bras qui bloque B, `clip10` | 0 | 0 | 0 | 0 | 0 |
| milieu, référence | 1 | 5 | 1 | 4 | 6 |
| milieu, référence + à-coups | 6 | 5 | 5 | 4 | 6 |
| milieu, `clip10` | 0 | 3 | 0 | 2 | 4 |
| milieu, `clip10` + à-coups | 1 | 4 | 3 | 6 | 6 |
| milieu, `cfix12` | 0 | 0 | 0 | 0 | 3 |
| milieu, `cfix12` + à-coups | 2 | 2 | 2 | 1 | 2 |

Ce qu'on observe :

- **Bras qui bloque B : les conclusions des 10 premiers runs tiennent sur 30.**
  - `clip10` ne touche jamais et reste à au moins 2,5 cm du bras.
  - La référence touche 3 fois, toujours à 0,5 Hz et de justesse : 6 échantillons au total, au pire −1,2 cm.
  - Sur 678 gestes, le robot n'arrive jamais à B tant que la main y est. Il attend à ~16 cm, côté base (x ≈ −0,33), puis atteint B ~2 s après le recul. Il attend, il ne contourne pas.
- **Bras au milieu : les à-coups augmentent nettement les contacts** pour les trois configs (17 → 26, 9 → 20 et 3 → 9 sur 30). Le classement ne change pas : `cfix12` > `clip10` > référence.
- **43 % des épisodes de contact commencent pendant un à-coup** (48 sur 111), contre 9 % attendus par hasard. Ce taux de hasard vient des runs sans à-coups, avec les mêmes instants pris comme repère : 6 sur 68. Avec `cfix12`, c'est 9 épisodes sur 13 : sans les à-coups, il resterait 2 runs sur 30 avec contact.
- **Avec `clip10`, les contacts dus aux à-coups ne sont plus forcément superficiels** : sur 30 runs, jusqu'à −11,0 cm. Sur 10 runs, ils restaient sous 1,6 cm.
- **`cfix12` à 0,5 Hz se bloque près de B, avec ou sans à-coups** : 3,7 à 4 B par run, et une cible abandonnée dans presque tous les runs.
  - La pince reste à 9–14 cm de B, avec q2 en butée haute (1,76 rad) et q1 vers −2,5 rad ; le bras humain est à 6–8 cm.
  - C'est le blocage de posture près de B déjà vu.
  - Aux autres fréquences, `cfix12` sans à-coups ne touche jamais (0/24).

Rapport mis à jour, avec 4 vidéos des nouveaux runs : https://claude.ai/artifact/4F8NKA4CzeshpjHK7qBsCH (section « Sur 30 runs par config »).

## 5. Modifications du code

Toutes les nouveautés sont **désactivées par défaut**. L'ancien code est commenté, pas supprimé.

| Fichier | Ajout |
|---|---|
| `ros2_ws/src/storm_pick_place_demo/scripts/urdf_human` | Paramètres `block_*`, en mode périodique ou réactif : articulations, postures repliée/tendue, durées, point et seuils de déclenchement, période réfractaire, repère TF du robot. Paramètres `jerk_*` et `forecast_knows_jerks`. Étiquette RViz « _main:B » / « _rapide ». Les segments en mouvement (bras bloquant, articulation à à-coups) sont publiés comme capsules dynamiques. |
| `ros2_ws/src/storm_pick_place_demo/storm_pick_place_demo/pick_place_task_manager_node.py` | `goal_block_distance` (0 = désactivé), `goal_block_release`, `goal_block_clear_sec`. C'est la règle d'arrêt de la version a. **Non utilisée** : l'arrêt doit venir de MPPI. |
| `research/campagnes_2026-11/wk.py` | Script de campagne. À chaque run, il réécrit le YAML MPPI, `urdf_human` et le gestionnaire de tâches depuis des sauvegardes, selon les flags (séparés par `+`), puis restaure tout à la fin. Copié depuis l'espace de travail de la session : les chemins de sauvegarde pointent encore vers ce dossier temporaire. |
| `research/campagnes_2026-11/m2b6_report.py`, `react_logfix.py` | Analyse : métriques par run, puis reconstruction des gestes réactifs. Les heures de départ viennent du journal de `urdf_human`, recalées sur le temps simulé. |
| `research/campagnes_2026-11/m4_report.py` | Analyse de wkM4 et wkB7 sur 30 runs par config : métriques par run, attribution des contacts aux à-coups, gestes réactifs (heures du journal recalées sur le temps simulé). Utilise `test_next.py` pour instancier `urdf_human` sans ROS. |
| `research/campagnes_2026-11/m3_report.py`, `m3_jerk2.py` | Analyse de wkM3 : métriques par run (wkM3 et runs appariés wkM1/wkM2), puis attribution de chaque épisode de contact à un à-coup ou non. Les instants des à-coups sont recalculés en instanciant `urdf_human` avec la même graine. |
| `research/campagnes_2026-11/chain*.sh` | Les commandes exactes des campagnes du 23 novembre. |

**Flags de scène :**

| Flag | Effet |
|---|---|
| `x100` | humain à x −1,00 |
| `y70` | humain décalé en y 0,70 |
| `alt` | amplitude alternée 0,15 / 0,45 |
| `mid` | épaule centrée à 0,9 rad |
| `blk` + `blkR` | bras droit qui avance sur B |
| `react` | déclenchement du geste à l'approche du robot |
| `jrk` | à-coups aléatoires |

**Flags MPPI :** `cattab`, `b75`, `pmax15`, `pmax20`, `clip10`, `cfix12`, `core3` (zone dure, non utilisée dans les campagnes du 23 novembre).

**Exemple de commande** (dans `cosmik_mppi_sim`, avec `ROS_LOCALHOST_ONLY=1`, `DEMO_LOCALHOST_ONLY=1`, `DEMO_DOMAIN_ID=93`) :

```bash
python3 -u research/campagnes_2026-11/wk.py repeat-wkB6 20261171 2 0.1,0.2,0.3,0.4,0.5 \
  react_clip10=x100+y70+cattab+b75+blk+blkR+react+jrk+pmax15+clip10
```

Les résultats sont écrits dans `.runtime/<campagne>/<config>/f<fréquence>/rep<n>/freq_*/` : `result.json`, `trace.npy` (temps, pince, distance au bras), `joints.npy`, `launch.log` et `video.mp4` (RViz).

## 6. Limites constatées

- **Bras au milieu du trajet à 0,5 Hz :** aucun réglage existant n'y atteint zéro contact. Avec les à-coups, aucun réglage n'atteint zéro contact sur l'ensemble des fréquences (au mieux 4 runs sur 10 avec `cfix12`).
  - Les contacts ont lieu en transit, souvent avec le robot bloqué sous l'avant-bras.
  - Des analyses antérieures montrent que le robot est alors souvent en butée de l'articulation q2 : il n'a plus de place pour s'écarter.
- **Bras réactif :** le robot attend mais ne contourne jamais la main. Contourner, par-dessus ou par-dessous, ne s'est jamais présenté comme la meilleure option pour MPPI dans cette scène.
- **Prévision :** elle reste analytique et connaît la forme du mouvement. Elle ne voit ni le départ des gestes ni les à-coups ; elle ne réagit qu'à l'instant présent. Le modèle de diffusion entraîné sur COMFI n'est pas encore branché.
- **Échantillon :** 2 runs par fréquence et par config dans les § 2, § 3 et § 4.3 ; 6 runs par fréquence (30 par config) dans le § 4.4. Les conclusions du § 4.4 remplacent celles tirées sur 10 runs quand elles diffèrent.

## 7. Rapports HTML (avec vidéos RViz)

Ces pages sont privées : elles doivent être partagées depuis leur menu « Share » pour être lisibles par un autre compte.

| Date | Rapport | Lien |
|---|---|---|
| 17 nov. | Bilan MPPI au 17 novembre | https://claude.ai/artifact/7zYCNM4tvd55665DFKN5Q6 |
| 20 nov. | Robot face aux deux amplitudes (x −1,10) | https://claude.ai/artifact/HXuwmFv2DFELXCrXuMn4mS |
| 20 nov. | Humain à x −1,00 (rapprochement, main gauche sur B) | https://claude.ai/artifact/Q6QCYC65YM51vof5EzRWyN |
| 23 nov. | Aperçu bras au milieu et main sur B | https://claude.ai/artifact/4srCgaigp5K1Y5Msr1MLck |
| 23 nov. | Bras au milieu, bras réactif (références) | https://claude.ai/artifact/Bqb7rNNqH9P9MkFQSzLqQ5 |
| 23 nov. | Réglages MPPI sur les deux scènes, avec la section mouvements rapides du bras au milieu (wkM3) | https://claude.ai/artifact/4F8NKA4CzeshpjHK7qBsCH |
| 23 nov. | Mouvements rapides aléatoires, bras au milieu (wkM3) | https://claude.ai/artifact/HYLHgnazsdpZngt7a6Wu2q |
