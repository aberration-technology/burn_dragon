use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::config::UsizeRangeConfig;
use crate::ruliad::config::{RuliadFamilyConfig, RuliadTaskKind};
use crate::ruliad::rng::SplitMix64;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadCategoryMorphism {
    pub name: String,
    pub source: usize,
    pub target: usize,
    pub identity: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadCategoryFunctor {
    pub name: String,
    pub object_map: Vec<usize>,
    pub morphism_map: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadNaturalityCheck {
    pub source_morphism: usize,
    pub left_path: Vec<usize>,
    pub right_path: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GeneratedCategorySpec {
    pub object_count: usize,
    pub morphisms: Vec<RuliadCategoryMorphism>,
    pub identities: Vec<usize>,
    pub composition: Vec<Vec<Option<usize>>>,
    pub path: Vec<usize>,
    pub composed: usize,
    pub lhs: usize,
    pub rhs: usize,
    pub holds: bool,
    pub proof_steps: Vec<String>,
    pub functor: Option<RuliadCategoryFunctor>,
    pub naturality: Option<RuliadNaturalityCheck>,
    pub task: RuliadTaskKind,
}

#[derive(Debug, Clone)]
struct GeneratedThinCategory {
    morphisms: Vec<RuliadCategoryMorphism>,
    identities: Vec<usize>,
    composition: Vec<Vec<Option<usize>>>,
    morphism_id: Vec<Vec<Option<usize>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GeneratedCategoryTaskFields {
    lhs: usize,
    rhs: usize,
    proof_steps: Vec<String>,
    functor: Option<RuliadCategoryFunctor>,
    naturality: Option<RuliadNaturalityCheck>,
}

pub(crate) fn generate_category_fields(
    family: &RuliadFamilyConfig,
    task: RuliadTaskKind,
    rng: &mut SplitMix64,
) -> Result<GeneratedCategorySpec> {
    let object_count = range_or(family.width, 3, 7, rng);
    let category = generated_thin_category(object_count);
    let selected_task = match task {
        RuliadTaskKind::ComposeCategoryPath
        | RuliadTaskKind::VerifyCategoryLaw
        | RuliadTaskKind::VerifyFunctorPreservation
        | RuliadTaskKind::VerifyNaturalitySquare => task,
        _ => RuliadTaskKind::ComposeCategoryPath,
    };
    let path_len = range_or(family.steps, 3, 6, rng).max(2);
    let object_path = monotone_object_path(object_count, path_len, rng);
    let mut path = object_path
        .windows(2)
        .map(|pair| category.morphism_id[pair[0]][pair[1]].expect("thin arrow"))
        .collect::<Vec<_>>();
    if path.is_empty() {
        path.push(category.identities[object_path[0]]);
    }
    let composed = compose_path(&category.morphisms, &category.composition, &path)
        .ok_or_else(|| anyhow!("generated invalid category path"))?;
    let task_fields =
        generated_category_task_fields(selected_task, &category, &path, composed, rng)?;

    Ok(GeneratedCategorySpec {
        object_count,
        morphisms: category.morphisms,
        identities: category.identities,
        composition: category.composition,
        path,
        composed,
        lhs: task_fields.lhs,
        rhs: task_fields.rhs,
        holds: task_fields.lhs == task_fields.rhs,
        proof_steps: task_fields.proof_steps,
        functor: task_fields.functor,
        naturality: task_fields.naturality,
        task: selected_task,
    })
}

fn generated_category_task_fields(
    task: RuliadTaskKind,
    category: &GeneratedThinCategory,
    path: &[usize],
    composed: usize,
    rng: &mut SplitMix64,
) -> Result<GeneratedCategoryTaskFields> {
    match task {
        RuliadTaskKind::ComposeCategoryPath => Ok(GeneratedCategoryTaskFields {
            lhs: composed,
            rhs: composed,
            proof_steps: category_path_proof_steps(
                &category.morphisms,
                &category.composition,
                path,
            ),
            functor: None,
            naturality: None,
        }),
        RuliadTaskKind::VerifyCategoryLaw => {
            let triple = composable_triple(category, rng);
            let fg = compose_pair(&category.composition, triple[0], triple[1])
                .ok_or_else(|| anyhow!("generated invalid category law lhs"))?;
            let gh = compose_pair(&category.composition, triple[1], triple[2])
                .ok_or_else(|| anyhow!("generated invalid category law rhs"))?;
            let lhs = compose_pair(&category.composition, fg, triple[2])
                .ok_or_else(|| anyhow!("generated invalid category law lhs fold"))?;
            let rhs = compose_pair(&category.composition, triple[0], gh)
                .ok_or_else(|| anyhow!("generated invalid category law rhs fold"))?;
            Ok(GeneratedCategoryTaskFields {
                lhs,
                rhs,
                proof_steps: vec![
                    format!(
                        "assoc_left:{} then {}",
                        morphism_name(&category.morphisms, fg),
                        morphism_name(&category.morphisms, lhs)
                    ),
                    format!(
                        "assoc_right:{} then {}",
                        morphism_name(&category.morphisms, gh),
                        morphism_name(&category.morphisms, rhs)
                    ),
                ],
                functor: None,
                naturality: None,
            })
        }
        RuliadTaskKind::VerifyFunctorPreservation => {
            let functor = generated_shift_functor(category, rng);
            let first = path[0];
            let second = path.get(1).copied().unwrap_or(first);
            let composed = compose_pair(&category.composition, first, second)
                .ok_or_else(|| anyhow!("generated invalid functor path"))?;
            let lhs = functor.morphism_map[composed];
            let rhs = compose_pair(
                &category.composition,
                functor.morphism_map[first],
                functor.morphism_map[second],
            )
            .ok_or_else(|| anyhow!("generated invalid functor composition"))?;
            Ok(GeneratedCategoryTaskFields {
                lhs,
                rhs,
                proof_steps: vec![
                    format!(
                        "F({}*{})={}",
                        morphism_name(&category.morphisms, first),
                        morphism_name(&category.morphisms, second),
                        morphism_name(&category.morphisms, lhs)
                    ),
                    format!(
                        "F({})*F({})={}",
                        morphism_name(&category.morphisms, first),
                        morphism_name(&category.morphisms, second),
                        morphism_name(&category.morphisms, rhs)
                    ),
                ],
                functor: Some(functor),
                naturality: None,
            })
        }
        RuliadTaskKind::VerifyNaturalitySquare => {
            let functor = generated_shift_functor(category, rng);
            let components = (0..category.identities.len())
                .map(|object| {
                    category.morphism_id[object][functor.object_map[object]]
                        .expect("naturality component")
                })
                .collect::<Vec<_>>();
            let source_morphism = *path.first().unwrap_or(&category.identities[0]);
            let source = category.morphisms[source_morphism].source;
            let target = category.morphisms[source_morphism].target;
            let left_path = vec![components[source], functor.morphism_map[source_morphism]];
            let right_path = vec![source_morphism, components[target]];
            let lhs = compose_path(&category.morphisms, &category.composition, &left_path)
                .ok_or_else(|| anyhow!("generated invalid naturality left path"))?;
            let rhs = compose_path(&category.morphisms, &category.composition, &right_path)
                .ok_or_else(|| anyhow!("generated invalid naturality right path"))?;
            Ok(GeneratedCategoryTaskFields {
                lhs,
                rhs,
                proof_steps: vec![
                    format!(
                        "left=F({}) after eta_o{}={}",
                        morphism_name(&category.morphisms, source_morphism),
                        source,
                        morphism_name(&category.morphisms, lhs)
                    ),
                    format!(
                        "right=eta_o{} after {}={}",
                        target,
                        morphism_name(&category.morphisms, source_morphism),
                        morphism_name(&category.morphisms, rhs)
                    ),
                ],
                functor: Some(functor),
                naturality: Some(RuliadNaturalityCheck {
                    source_morphism,
                    left_path,
                    right_path,
                }),
            })
        }
        _ => unreachable!("category task normalized before generation"),
    }
}

fn generated_thin_category(object_count: usize) -> GeneratedThinCategory {
    let mut morphisms = Vec::new();
    let mut morphism_id = vec![vec![None; object_count]; object_count];
    for (source, row) in morphism_id.iter_mut().enumerate() {
        for (target, slot) in row.iter_mut().enumerate().skip(source) {
            let id = morphisms.len();
            *slot = Some(id);
            morphisms.push(RuliadCategoryMorphism {
                name: format!("m{source}_{target}"),
                source,
                target,
                identity: source == target,
            });
        }
    }
    let identities = morphism_id
        .iter()
        .enumerate()
        .map(|(object, row)| row[object].expect("identity"))
        .collect::<Vec<_>>();
    let mut composition = vec![vec![None; morphisms.len()]; morphisms.len()];
    for (left_id, left) in morphisms.iter().enumerate() {
        for (right_id, right) in morphisms.iter().enumerate() {
            if left.target == right.source {
                composition[left_id][right_id] = morphism_id[left.source][right.target];
            }
        }
    }
    GeneratedThinCategory {
        morphisms,
        identities,
        composition,
        morphism_id,
    }
}

fn monotone_object_path(object_count: usize, path_len: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut path = Vec::with_capacity(path_len.max(2));
    let mut current = rng.next_usize(object_count);
    path.push(current);
    for _ in 1..path_len.max(2) {
        let remaining = object_count.saturating_sub(current + 1);
        current += rng.next_usize(remaining + 1);
        path.push(current);
    }
    path
}

fn composable_triple(category: &GeneratedThinCategory, rng: &mut SplitMix64) -> [usize; 3] {
    let objects = monotone_object_path(category.identities.len(), 4, rng);
    [
        category.morphism_id[objects[0]][objects[1]].expect("first arrow"),
        category.morphism_id[objects[1]][objects[2]].expect("second arrow"),
        category.morphism_id[objects[2]][objects[3]].expect("third arrow"),
    ]
}

fn generated_shift_functor(
    category: &GeneratedThinCategory,
    rng: &mut SplitMix64,
) -> RuliadCategoryFunctor {
    let object_count = category.identities.len();
    let shift = rng.range_usize(0, object_count.saturating_sub(1));
    let object_map = (0..object_count)
        .map(|object| object.saturating_add(shift).min(object_count - 1))
        .collect::<Vec<_>>();
    let morphism_map = category
        .morphisms
        .iter()
        .map(|morphism| {
            category.morphism_id[object_map[morphism.source]][object_map[morphism.target]]
                .expect("functor arrow")
        })
        .collect::<Vec<_>>();
    RuliadCategoryFunctor {
        name: format!("shift_{shift}"),
        object_map,
        morphism_map,
    }
}

pub(crate) fn valid_finite_category(
    object_count: usize,
    morphisms: &[RuliadCategoryMorphism],
    identities: &[usize],
    composition: &[Vec<Option<usize>>],
) -> bool {
    if object_count == 0
        || morphisms.is_empty()
        || identities.len() != object_count
        || composition.len() != morphisms.len()
        || composition.iter().any(|row| row.len() != morphisms.len())
    {
        return false;
    }
    for (id, morphism) in morphisms.iter().enumerate() {
        if morphism.source >= object_count
            || morphism.target >= object_count
            || morphism.name.trim().is_empty()
            || morphism.name.chars().any(char::is_whitespace)
        {
            return false;
        }
        if morphism.identity
            && (morphism.source != morphism.target
                || identities.get(morphism.source).copied() != Some(id))
        {
            return false;
        }
    }
    for (object, identity) in identities.iter().copied().enumerate() {
        let Some(morphism) = morphisms.get(identity) else {
            return false;
        };
        if !morphism.identity || morphism.source != object || morphism.target != object {
            return false;
        }
    }
    for (left_id, left) in morphisms.iter().enumerate() {
        for (right_id, right) in morphisms.iter().enumerate() {
            let composed = composition[left_id][right_id];
            if left.target == right.source {
                let Some(composed_id) = composed else {
                    return false;
                };
                let Some(result) = morphisms.get(composed_id) else {
                    return false;
                };
                if result.source != left.source || result.target != right.target {
                    return false;
                }
            } else if composed.is_some() {
                return false;
            }
        }
    }
    for (morphism_id, morphism) in morphisms.iter().enumerate() {
        if compose_pair(composition, identities[morphism.source], morphism_id) != Some(morphism_id)
            || compose_pair(composition, morphism_id, identities[morphism.target])
                != Some(morphism_id)
        {
            return false;
        }
    }
    for (first_id, first) in morphisms.iter().enumerate() {
        for (second_id, second) in morphisms.iter().enumerate() {
            for (third_id, third) in morphisms.iter().enumerate() {
                if first.target != second.source || second.target != third.source {
                    continue;
                }
                let left = compose_pair(composition, first_id, second_id)
                    .and_then(|partial| compose_pair(composition, partial, third_id));
                let right = compose_pair(composition, second_id, third_id)
                    .and_then(|partial| compose_pair(composition, first_id, partial));
                if left.is_none() || left != right {
                    return false;
                }
            }
        }
    }
    true
}

pub(crate) fn valid_functor(
    object_count: usize,
    morphisms: &[RuliadCategoryMorphism],
    identities: &[usize],
    composition: &[Vec<Option<usize>>],
    functor: &RuliadCategoryFunctor,
) -> bool {
    if object_count == 0
        || identities.len() != object_count
        || functor.object_map.len() != object_count
        || functor.morphism_map.len() != morphisms.len()
        || functor
            .object_map
            .iter()
            .any(|object| *object >= object_count)
        || functor
            .morphism_map
            .iter()
            .any(|morphism| *morphism >= morphisms.len())
        || morphisms
            .iter()
            .any(|morphism| morphism.source >= object_count || morphism.target >= object_count)
    {
        return false;
    }
    for (source_id, source) in morphisms.iter().enumerate() {
        let Some(mapped) = morphisms.get(functor.morphism_map[source_id]) else {
            return false;
        };
        if mapped.source != functor.object_map[source.source]
            || mapped.target != functor.object_map[source.target]
        {
            return false;
        }
    }
    for (object, identity) in identities.iter().copied().enumerate() {
        if identity >= functor.morphism_map.len() {
            return false;
        }
        let mapped_object = functor.object_map[object];
        if functor.morphism_map[identity] != identities[mapped_object] {
            return false;
        }
    }
    for (left_id, row) in composition.iter().enumerate() {
        for (right_id, composed) in row.iter().enumerate() {
            let Some(composed) = composed else {
                continue;
            };
            let lhs = functor.morphism_map[*composed];
            let rhs = compose_pair(
                composition,
                functor.morphism_map[left_id],
                functor.morphism_map[right_id],
            );
            if rhs != Some(lhs) {
                return false;
            }
        }
    }
    true
}

pub(crate) fn naturality_commutes(
    morphisms: &[RuliadCategoryMorphism],
    composition: &[Vec<Option<usize>>],
    functor: &RuliadCategoryFunctor,
    naturality: &RuliadNaturalityCheck,
) -> bool {
    if naturality.source_morphism >= morphisms.len()
        || naturality.left_path.len() != 2
        || naturality.right_path.len() != 2
        || functor.object_map.is_empty()
        || functor.morphism_map.len() != morphisms.len()
        || functor
            .morphism_map
            .iter()
            .any(|morphism| *morphism >= morphisms.len())
        || functor
            .object_map
            .iter()
            .any(|object| *object >= functor.object_map.len())
    {
        return false;
    }
    let source_morphism = &morphisms[naturality.source_morphism];
    if source_morphism.source >= functor.object_map.len()
        || source_morphism.target >= functor.object_map.len()
        || naturality.left_path[1] != functor.morphism_map[naturality.source_morphism]
        || naturality.right_path[0] != naturality.source_morphism
    {
        return false;
    }
    let left = compose_path(morphisms, composition, &naturality.left_path);
    let right = compose_path(morphisms, composition, &naturality.right_path);
    let Some(left_id) = left else {
        return false;
    };
    let Some(right_id) = right else {
        return false;
    };
    let Some(left_first) = naturality
        .left_path
        .first()
        .and_then(|id| morphisms.get(*id))
    else {
        return false;
    };
    let Some(left_last) = naturality
        .left_path
        .last()
        .and_then(|id| morphisms.get(*id))
    else {
        return false;
    };
    let Some(right_first) = naturality
        .right_path
        .first()
        .and_then(|id| morphisms.get(*id))
    else {
        return false;
    };
    let Some(right_last) = naturality
        .right_path
        .last()
        .and_then(|id| morphisms.get(*id))
    else {
        return false;
    };
    left_id == right_id
        && left_first.source == source_morphism.source
        && left_first.target == functor.object_map[source_morphism.source]
        && left_last.source == functor.object_map[source_morphism.source]
        && left_last.target == functor.object_map[source_morphism.target]
        && right_first.source == source_morphism.source
        && right_first.target == source_morphism.target
        && right_last.source == source_morphism.target
        && right_last.target == functor.object_map[source_morphism.target]
}

pub(crate) fn compose_path(
    morphisms: &[RuliadCategoryMorphism],
    composition: &[Vec<Option<usize>>],
    path: &[usize],
) -> Option<usize> {
    let mut current = *path.first()?;
    if current >= morphisms.len() {
        return None;
    }
    for next in &path[1..] {
        if *next >= morphisms.len() {
            return None;
        }
        current = compose_pair(composition, current, *next)?;
    }
    Some(current)
}

fn compose_pair(composition: &[Vec<Option<usize>>], left: usize, right: usize) -> Option<usize> {
    composition.get(left)?.get(right).copied().flatten()
}

fn category_path_proof_steps(
    morphisms: &[RuliadCategoryMorphism],
    composition: &[Vec<Option<usize>>],
    path: &[usize],
) -> Vec<String> {
    if path.is_empty() {
        return Vec::new();
    }
    let mut steps = Vec::new();
    let mut current = path[0];
    for next in &path[1..] {
        if let Some(composed) = compose_pair(composition, current, *next) {
            steps.push(format!(
                "{}*{}={}",
                morphism_name(morphisms, current),
                morphism_name(morphisms, *next),
                morphism_name(morphisms, composed)
            ));
            current = composed;
        }
    }
    steps
}

fn morphism_name(morphisms: &[RuliadCategoryMorphism], id: usize) -> String {
    morphisms
        .get(id)
        .map(|morphism| morphism.name.clone())
        .unwrap_or_else(|| format!("m?{id}"))
}

fn range_or(
    range: Option<UsizeRangeConfig>,
    default_min: usize,
    default_max: usize,
    rng: &mut SplitMix64,
) -> usize {
    match range {
        Some(range) => rng.range_usize(range.min, range.max),
        None => rng.range_usize(default_min, default_max),
    }
}
