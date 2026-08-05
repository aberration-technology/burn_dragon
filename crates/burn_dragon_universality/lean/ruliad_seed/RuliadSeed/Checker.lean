namespace RuliadSeed

def ruliadIrVersion : Nat := 3

inductive Term where
  | variable (index : Nat)
  | atom (symbol : String)
  | apply (operator : String) (arguments : List Term)
  deriving Repr, BEq

structure Equality where
  lhs : Term
  rhs : Term
  deriving Repr, BEq

structure RewriteAxiom where
  id : String
  lhs : Term
  rhs : Term
  deriving Repr, BEq

structure ProofGoal where
  dependencies : List Nat
  claim : Equality
  deriving Repr, BEq

structure Problem where
  version : Nat
  axioms : List RewriteAxiom
  goals : List ProofGoal
  root : Nat
  deriving Repr, BEq

inductive RewriteDirection where
  | forward
  | reverse
  deriving Repr, BEq

inductive ProofSource where
  | namedAxiom (id : String)
  | priorGoal (goal : Nat)
  deriving Repr, BEq

structure ProofStep where
  source : ProofSource
  path : List Nat
  direction : RewriteDirection
  deriving Repr, BEq

structure GoalCertificate where
  goal : Nat
  steps : List ProofStep
  deriving Repr, BEq

structure Certificate where
  version : Nat
  goals : List GoalCertificate
  deriving Repr, BEq

abbrev Substitution := List (Nat × Term)

def lookupSubstitution (index : Nat) : Substitution → Option Term
  | [] => none
  | (key, value) :: rest =>
      if key == index then some value else lookupSubstitution index rest

mutual
  partial def matchTerm
      (pattern value : Term) (substitution : Substitution) : Option Substitution :=
    match pattern, value with
    | .variable index, value =>
        match lookupSubstitution index substitution with
        | some bound => if bound == value then some substitution else none
        | none => some ((index, value) :: substitution)
    | .atom left, .atom right =>
        if left == right then some substitution else none
    | .apply leftOperator leftArguments, .apply rightOperator rightArguments =>
        if leftOperator == rightOperator then
          matchTermList leftArguments rightArguments substitution
        else
          none
    | _, _ => none

  partial def matchTermList
      (patterns values : List Term) (substitution : Substitution) : Option Substitution :=
    match patterns, values with
    | [], [] => some substitution
    | pattern :: patternRest, value :: valueRest => do
        let next ← matchTerm pattern value substitution
        matchTermList patternRest valueRest next
    | _, _ => none
end

partial def instantiate (template : Term) (substitution : Substitution) : Option Term :=
  match template with
  | .variable index => lookupSubstitution index substitution
  | .atom symbol => some (.atom symbol)
  | .apply operator arguments => do
      let instantiated ← arguments.mapM (fun argument => instantiate argument substitution)
      some (.apply operator instantiated)

def replaceAt {α : Type} (values : List α) (index : Nat) (replacement : α) : Option (List α) :=
  match values, index with
  | [], _ => none
  | _ :: rest, 0 => some (replacement :: rest)
  | value :: rest, index + 1 => do
      let replaced ← replaceAt rest index replacement
      some (value :: replaced)

partial def rewriteAtPath
    (term : Term) (path : List Nat) (pattern replacement : Term) : Option Term :=
  match path with
  | [] => do
      let substitution ← matchTerm pattern term []
      instantiate replacement substitution
  | index :: rest =>
      match term with
      | .apply operator arguments => do
          let argument ← arguments[index]?
          let rewritten ← rewriteAtPath argument rest pattern replacement
          let nextArguments ← replaceAt arguments index rewritten
          some (.apply operator nextArguments)
      | _ => none

def findAxiom (axioms : List RewriteAxiom) (id : String) : Option RewriteAxiom :=
  axioms.find? (fun rule => rule.id == id)

def sourceEquality
    (problem : Problem) (goal : ProofGoal) (source : ProofSource) : Option Equality :=
  match source with
  | .namedAxiom id => do
      let rule ← findAxiom problem.axioms id
      some { lhs := rule.lhs, rhs := rule.rhs }
  | .priorGoal dependency =>
      if goal.dependencies.contains dependency then
        problem.goals[dependency]? |>.map (fun dependencyGoal => dependencyGoal.claim)
      else
        none

def applyStep
    (problem : Problem) (goal : ProofGoal) (current : Term) (step : ProofStep) : Option Term := do
  let equality ← sourceEquality problem goal step.source
  let (pattern, replacement) :=
    match step.direction with
    | .forward => (equality.lhs, equality.rhs)
    | .reverse => (equality.rhs, equality.lhs)
  rewriteAtPath current step.path pattern replacement

def replaySteps
    (problem : Problem) (goal : ProofGoal) : List ProofStep → Term → Option Term
  | [], current => some current
  | step :: rest, current => do
      let next ← applyStep problem goal current step
      replaySteps problem goal rest next

def reachesGoal (problem : Problem) (target current : Nat) : Nat → Bool
  | 0 => false
  | fuel + 1 =>
      if target == current then
        true
      else
        match problem.goals[current]? with
        | none => false
        | some goal => goal.dependencies.any (fun dependency =>
            reachesGoal problem target dependency fuel)

def requiredGoals (problem : Problem) : List Nat :=
  List.range problem.goals.length |>.filter (fun target =>
    reachesGoal problem target problem.root (problem.goals.length + 1))

def findGoalCertificate
    (certificates : List GoalCertificate) (goal : Nat) : Option GoalCertificate :=
  certificates.find? (fun certificate => certificate.goal == goal)

def verifyGoalSequence
    (problem : Problem) (certificate : Certificate) : List Nat → List Nat → Bool
  | [], _ => true
  | goalIndex :: rest, verified =>
      match problem.goals[goalIndex]?, findGoalCertificate certificate.goals goalIndex with
      | some goal, some goalCertificate =>
          goal.dependencies.all (fun dependency => verified.contains dependency) &&
          (replaySteps problem goal goalCertificate.steps goal.claim.lhs == some goal.claim.rhs) &&
          verifyGoalSequence problem certificate rest (goalIndex :: verified)
      | _, _ => false

def checkCertificate (problem : Problem) (certificate : Certificate) : Bool :=
  let required := requiredGoals problem
  problem.version == ruliadIrVersion &&
    certificate.version == ruliadIrVersion &&
    !required.isEmpty &&
    problem.root < problem.goals.length &&
    certificate.goals.map (fun goal => goal.goal) == required &&
    verifyGoalSequence problem certificate required []

private def fixtureProblem : Problem := {
  version := ruliadIrVersion
  axioms := [{
    id := "drop"
    lhs := .apply "wrap" [.variable 0]
    rhs := .variable 0
  }]
  goals := [
    {
      dependencies := []
      claim := {
        lhs := .apply "wrap" [.atom "a"]
        rhs := .atom "a"
      }
    },
    {
      dependencies := [0]
      claim := {
        lhs := .apply "pair" [.apply "wrap" [.atom "a"], .atom "b"]
        rhs := .apply "pair" [.atom "a", .atom "b"]
      }
    },
    {
      dependencies := [1]
      claim := {
        lhs := .atom "a"
        rhs := .apply "wrap" [.atom "a"]
      }
    }
  ]
  root := 2
}

private def fixtureCertificate : Certificate := {
  version := ruliadIrVersion
  goals := [
    {
      goal := 0
      steps := [{ source := .namedAxiom "drop", path := [], direction := .forward }]
    },
    {
      goal := 1
      steps := [{ source := .priorGoal 0, path := [0], direction := .forward }]
    },
    {
      goal := 2
      steps := [{ source := .namedAxiom "drop", path := [], direction := .reverse }]
    }
  ]
}

example : checkCertificate fixtureProblem fixtureCertificate = true := by
  native_decide

example : checkCertificate fixtureProblem {
    fixtureCertificate with goals := fixtureCertificate.goals.dropLast
  } = false := by
  native_decide

example : checkCertificate fixtureProblem {
    fixtureCertificate with goals := fixtureCertificate.goals.map fun goal =>
      if goal.goal == 0 then
        { goal with steps := [{ source := .namedAxiom "missing", path := [], direction := .forward }] }
      else goal
  } = false := by
  native_decide

end RuliadSeed
