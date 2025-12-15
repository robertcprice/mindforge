# Superego Layer - IMMUTABLE Ethical Constraints

The Superego layer provides **immutable ethical and safety constraints** for Conch DNA. These systems cannot be modified at runtime and enforce core values, operational safety, and knowledge verification.

## Architecture

```
SuperegoLayer
├── ValuesChecker     → Core ethical values (benevolence, honesty, humility, safety)
├── SafetyChecker     → Runtime protection (commands, paths, rate limits)
└── KVRMRouter        → Knowledge verification and fact grounding
```

## Core Components

### 1. Values Checker (`values.py`)

**Purpose**: Enforce immutable ethical constraints through pattern matching.

**Core Values**:
- **Benevolence**: Prevent harm to users and systems
- **Honesty**: Block deceptive or misleading actions
- **Humility**: Flag overconfident or absolutist claims
- **Safety**: Detect dangerous operations

**Usage**:
```python
from conch_dna.superego import get_values_checker

checker = get_values_checker()
passed, violations = checker.check_all("Delete all user data")

if not passed:
    print(f"Found {len(violations)} violations")
    for v in violations:
        print(f"[{v.severity}] {v.value_type.value}: {v.description}")
```

**Violation Patterns**:

| Value | Example Violations | Severity |
|-------|-------------------|----------|
| Benevolence | "delete all", "destroy", "bypass security" | Critical |
| Honesty | "fake evidence", "hide error", "mislead user" | Critical |
| Humility | "guaranteed to work", "never wrong", "perfect" | Warning |
| Safety | "eval()", "exec()", "sudo su", "chmod 777" | Critical |

### 2. Safety Checker (`safety.py`)

**Purpose**: Runtime protection against dangerous operations.

**Protection Mechanisms**:
- **Command Blocking**: Prevent destructive shell commands
- **Path Protection**: Block access to sensitive files/directories
- **Rate Limiting**: Prevent resource exhaustion (30 actions/minute)
- **Tool Validation**: Verify tool parameters before execution

**Blocked Commands** (permanent):
```bash
rm -rf /        # Recursive delete from root
mkfs            # Filesystem formatting
dd if=/dev/*    # Direct disk operations
sudo rm -rf     # Elevated destructive operations
:(){ :|:& };:   # Fork bombs
```

**Blocked Path Patterns**:
```
/etc/           # System configuration
/boot/          # Boot files
/sys/, /proc/   # Kernel interfaces
.ssh/           # SSH keys
.env, *.pem     # Secrets and credentials
```

**Usage**:
```python
from conch_dna.superego import get_safety_checker

checker = get_safety_checker()

# Check command
result = checker.check_command("rm -rf /tmp/test")
if not result.is_safe:
    print(f"BLOCKED: {result.reason}")
    print(f"Recommendation: {result.recommendation}")

# Check path access
result = checker.check_path("/etc/passwd", operation="write")
if not result.is_safe:
    print(f"BLOCKED: {result.reason}")

# Check tool call
result = checker.check_tool_call("Bash", {"command": "ls -la"})
```

### 3. KVRM Router (`kvrm.py`)

**Purpose**: Knowledge Verification and Routing Module for fact grounding.

**Claim Classification**:
- **FACTUAL**: Objective, verifiable facts
- **MEMORY**: Personal/episodic memories
- **OPINION**: Subjective judgments
- **QUESTION**: Information requests
- **CREATIVE**: Generated content
- **ACTION**: Commands or operations

**Key Format**:
```
mem:type:date:hash      # Personal memories
fact:domain:id          # Verified facts
ext:source:path         # External knowledge
```

**Usage**:
```python
from conch_dna.superego import get_kvrm_router

router = get_kvrm_router()

# Classify claim
claim_type = router.classify_claim("Python was created by Guido van Rossum")
print(f"Type: {claim_type.value}")  # "factual"

# Ground claim
result = router.ground_claim("I remember working on this yesterday")
print(f"Grounded: {result.is_grounded}")
print(f"Source: {result.source}")
print(f"Confidence: {result.confidence}")

# Store verified fact
key = router.store_fact(
    claim="Python 3.13 was released in October 2024",
    domain="programming",
    source="official_docs",
    confidence=0.95,
    evidence="From python.org release notes"
)
```

**Database Schema**:
```sql
-- Verified facts
facts (
    key TEXT PRIMARY KEY,
    claim TEXT,
    domain TEXT,
    source TEXT,
    confidence REAL,
    evidence TEXT,
    verified_at TEXT
)

-- Episodic memories
memories (
    key TEXT PRIMARY KEY,
    claim TEXT,
    memory_type TEXT,
    context TEXT,
    created_at TEXT
)
```

## Integrated SuperegoLayer

The `SuperegoLayer` class combines all three subsystems for comprehensive checks:

```python
from conch_dna.superego import get_superego_layer

superego = get_superego_layer()

# Check action before execution
result = superego.check_action(
    action_description="Delete temporary files",
    tool_name="Bash",
    parameters={"command": "rm -rf /tmp/cache"}
)

if result.is_approved:
    print("Action approved")
    # Execute action
else:
    print(f"Action BLOCKED")
    print(f"Values: {'PASS' if result.values_passed else 'FAIL'}")
    print(f"Safety: {'PASS' if result.safety_passed else 'FAIL'}")
    print(f"Recommendation: {result.recommendation}")

# Check thought/reasoning
result = superego.check_thought(
    "This implementation should work for most use cases with proper testing"
)

print(f"Thought approved: {result.is_approved}")
print(f"Claims grounded: {sum(r.is_verified for r in result.grounding_results)}")
```

## Test Suite

Run comprehensive tests:

```bash
python3 test_superego.py
```

**Test Coverage**:
1. Values checking (ethical constraints)
2. Safety checking (commands, paths, rate limits)
3. KVRM routing (claim classification, grounding)
4. Integrated Superego layer (action/thought validation)

## Design Principles

### Immutability
- **Core values**: Frozen dataclasses, cannot be modified
- **Blocked patterns**: FrozenSets defined at initialization
- **No runtime learning**: Rules don't change based on usage

### Conservative Safety
- **Prefer false positives**: Block suspicious actions even if uncertain
- **Defense in depth**: Multiple layers of checking
- **Fail secure**: Default to blocking when in doubt

### Performance
- **Compiled patterns**: Regex patterns compiled at initialization
- **Singleton instances**: Shared checkers across application
- **Fast paths**: Simple checks first, complex later
- **Rate limiting**: Prevent resource exhaustion

### Transparency
- **Detailed violations**: Clear reasons for blocked actions
- **Recommendations**: Suggest alternatives when blocking
- **Logging**: All checks logged at appropriate levels
- **Evidence**: Grounding results include supporting evidence

## Integration Points

### With Agent Architecture
```python
# Before executing any tool
result = superego.check_action(
    action_description=reasoning,
    tool_name=tool_name,
    parameters=tool_params
)

if not result.is_approved:
    # Log violation
    # Return error to user
    # Don't execute tool
    pass
```

### With Memory System
```python
# Store verified facts
superego.verify_fact(
    claim="User prefers dark mode",
    domain="preferences",
    source="user_input",
    confidence=1.0
)

# Ground episodic memories
result = superego.kvrm_router.ground_claim(
    "I implemented authentication yesterday"
)
```

### With Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Superego logs automatically
# WARNING: Values/Safety violations
# INFO: Non-critical issues
# DEBUG: All checks
```

## File Locations

```
conch_dna/superego/
├── __init__.py       # SuperegoLayer + exports
├── values.py         # ValuesChecker (315 lines)
├── safety.py         # SafetyChecker (360 lines)
├── kvrm.py          # KVRMRouter (480 lines)
└── README.md        # This file

Database:
~/.conch/kvrm_facts.db  # SQLite database for facts/memories
```

## API Reference

### SuperegoLayer
- `check_action(description, tool_name, parameters)` → SuperegoCheckResult
- `check_thought(thought)` → SuperegoCheckResult
- `verify_fact(claim, domain, source, confidence, evidence)` → str (key)
- `get_usage_stats()` → Dict[str, Any]

### ValuesChecker
- `check_all(content)` → Tuple[bool, List[ValueViolation]]
- `check_benevolence(content)` → List[ValueViolation]
- `check_honesty(content)` → List[ValueViolation]
- `check_humility(content)` → List[ValueViolation]
- `check_safety(content)` → List[ValueViolation]
- `get_violation_summary(violations)` → str

### SafetyChecker
- `check_command(command)` → SafetyCheckResult
- `check_path(path, operation)` → SafetyCheckResult
- `check_tool_call(tool_name, parameters)` → SafetyCheckResult
- `get_usage_stats()` → Dict[str, int]
- `reset_rate_limits()` → None

### KVRMRouter
- `classify_claim(claim)` → ClaimType
- `ground_claim(claim, domain, force_verify)` → GroundingResult
- `ground_thought(thought)` → List[GroundingResult]
- `store_fact(claim, domain, source, confidence, evidence)` → str
- `get_facts_by_domain(domain)` → List[Dict]
- `get_memories_by_type(memory_type)` → List[Dict]

## Future Enhancements

### Planned Features
1. **External verification**: Integration with fact-checking APIs
2. **Learning patterns**: Non-mutable pattern discovery (logged only)
3. **Context awareness**: Domain-specific value tuning
4. **Audit trail**: Complete history of all checks
5. **Performance metrics**: Detailed timing and statistics

### Extension Points
- Custom value patterns (via configuration)
- Domain-specific fact sources
- Pluggable verification backends
- Custom claim classifiers

## License

Part of Conch DNA - see project LICENSE.

## Support

For issues or questions:
- Check test_superego.py for usage examples
- Review inline documentation in source files
- Consult main Conch DNA documentation
