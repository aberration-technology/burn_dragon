namespace RuliadSeed

universe u v w

def Step (A : Type u) := A -> A

def Commutes {A : Type u} {B : Type v}
    (stepA : Step A) (stepB : Step B) (map : A -> B) : Prop :=
  forall x, map (stepA x) = stepB (map x)

def iter {A : Type u} (step : Step A) : Nat -> A -> A
  | 0, x => x
  | n + 1, x => step (iter step n x)

theorem identity_simulation {A : Type u} (step : Step A) :
    Commutes step step id := by
  intro x
  rfl

theorem simulation_composition
    {A : Type u} {B : Type v} {C : Type w}
    {stepA : Step A} {stepB : Step B} {stepC : Step C}
    {f : A -> B} {g : B -> C}
    (hf : Commutes stepA stepB f)
    (hg : Commutes stepB stepC g) :
    Commutes stepA stepC (fun x => g (f x)) := by
  intro x
  rw [hf x, hg (f x)]

theorem finite_trajectory_preservation
    {A : Type u} {B : Type v}
    {stepA : Step A} {stepB : Step B} {map : A -> B}
    (h : Commutes stepA stepB map) :
    forall n x, map (iter stepA n x) = iter stepB n (map x)
  | 0, x => rfl
  | n + 1, x => by
      simp [iter, h (iter stepA n x), finite_trajectory_preservation h n x]

inductive Rewrites {A : Type u} (rel : A -> A -> Prop) : A -> A -> Prop
  | refl (x : A) : Rewrites rel x x
  | step {x y z : A} : rel x y -> Rewrites rel y z -> Rewrites rel x z

theorem rewrite_chain_composition
    {A : Type u} {rel : A -> A -> Prop} {x y z : A}
    (hxy : Rewrites rel x y) (hyz : Rewrites rel y z) :
    Rewrites rel x z := by
  induction hxy with
  | refl x => exact hyz
  | step hxy _ ih => exact Rewrites.step hxy (ih hyz)

end RuliadSeed
