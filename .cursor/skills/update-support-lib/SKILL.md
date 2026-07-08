# Update Support Library

Guide for modifying the shared test support library at `tests/common/support/`.

## File organization

| File | Domain |
|------|--------|
| `test.go` | Test interface (`With(t)`, `Ctx()`, `Client()`, `NewTestNamespace()`) |
| `client.go` | Client interface with 13 API accessors (Core, Trainer, Kubeflow, Ray, etc.) |
| `namespace.go` | Namespace creation, cleanup, log/event capture |
| `environment.go` | Environment variable constants and typed getter functions |
| `defaults.go` | Hardcoded default image versions and fallback values |
| `core.go` | Pod, ConfigMap, Secret, PVC helpers |
| `trainjob.go` | TrainJob getters and condition checkers |
| `pytorchjob.go` | PyTorchJob getters and condition checkers |
| `ray.go` | RayJob/RayCluster helpers |
| `kueue.go` | ResourceFlavor, ClusterQueue, LocalQueue helpers |
| `conditions.go` | Generic Kubernetes condition evaluation |
| `events.go` | Event capture and formatting for debugging |
| `rbac.go` | Role/RoleBinding creation |
| `accelerator.go` | GPU node detection |
| `fakeclient.go` | Fake client setup for unit tests (`NewTest(t)`) |

## Async getter pattern

Resource getters return a closure for use with `test.Eventually(...)`:

```go
func TrainJob(t Test, namespace, name string) func(g gomega.Gomega) *trainerv1alpha1.TrainJob {
    return func(g gomega.Gomega) *trainerv1alpha1.TrainJob {
        job, err := t.Client().Trainer().TrainerV1alpha1().TrainJobs(namespace).Get(t.Ctx(), name, metav1.GetOptions{})
        g.Expect(err).NotTo(gomega.HaveOccurred())
        return job
    }
}
```

Follow this pattern when adding getters for new resource types. The outer function captures the test context; the inner function is retried by gomega.

## Resource creation pattern

```go
func CreateMyResource(t Test, namespace string, content map[string][]byte) *corev1.MyResource {
    t.T().Helper()

    resource := &corev1.MyResource{
        TypeMeta: metav1.TypeMeta{
            APIVersion: corev1.SchemeGroupVersion.String(),
            Kind:       "MyResource",
        },
        ObjectMeta: metav1.ObjectMeta{
            GenerateName: "my-resource-",
            Namespace:    namespace,
        },
        // ... fields
    }

    resource, err := t.Client().Core().CoreV1().MyResources(namespace).Create(t.Ctx(), resource, metav1.CreateOptions{})
    t.Expect(err).NotTo(gomega.HaveOccurred())
    t.T().Logf("Created MyResource %s/%s successfully", namespace, resource.Name)

    return resource
}
```

Key conventions:
- Always call `t.T().Helper()` first
- Use `GenerateName`, never fixed `Name`
- Assert errors with `t.Expect(err).NotTo(gomega.HaveOccurred())`
- Log the created resource name

## Condition checker pattern

```go
func MyResourceConditionReady(resource *v1alpha1.MyResource) metav1.ConditionStatus {
    return MyResourceCondition(resource, v1alpha1.MyResourceReady)
}

func MyResourceCondition(resource *v1alpha1.MyResource, conditionType string) metav1.ConditionStatus {
    for _, condition := range resource.Status.Conditions {
        if string(condition.Type) == conditionType {
            return condition.Status
        }
    }
    return metav1.ConditionUnknown
}
```

Create one exported function per condition type (Ready, Failed, Complete, etc.) that delegates to a generic condition extractor.

## Option pattern

Used for flexible configuration of namespace, PVC, and other resources:

```go
type Option[T any] interface {
    ApplyTo(to T) error
}

type ErrorOption[T any] func(to T) error
func (f ErrorOption[T]) ApplyTo(to T) error { return f(to) }
```

Example - adding a label to a namespace:

```go
func WithKueueManaged() Option[*corev1.Namespace] {
    return ErrorOption[*corev1.Namespace](func(ns *corev1.Namespace) error {
        if ns.Labels == nil {
            ns.Labels = make(map[string]string)
        }
        ns.Labels["kueue.x-k8s.io/managed"] = "true"
        return nil
    })
}
```

Options are applied via a loop before the API call:

```go
for _, option := range options {
    t.Expect(option.ApplyTo(resource)).To(gomega.Succeed())
}
```

## Adding a new API client

To add a client for a new Kubernetes API:

1. **Add the import** in `client.go`:
   ```go
   newclient "github.com/org/project/pkg/client/clientset/versioned"
   ```

2. **Extend the `Client` interface**:
   ```go
   NewAPI() newclient.Interface
   ```

3. **Add a field to `testClient` struct**:
   ```go
   newAPI newclient.Interface
   ```

4. **Add the accessor method**:
   ```go
   func (t *testClient) NewAPI() newclient.Interface { return t.newAPI }
   ```

5. **Initialize in `newTestClient()`** (in `test.go`):
   ```go
   newAPI, err := newclient.NewForConfig(cfg)
   // handle error
   ```

6. **Update `fakeclient.go`** to include the new client for unit tests.

7. **Run `go mod tidy`** to pull the new dependency.

## Adding environment variables

Follow the constant + getter pattern in `environment.go`:

```go
const (
    MyNewVar = "MY_NEW_VAR"
)

func GetMyNewVar(t Test) string {
    t.T().Helper()
    return lookupEnvOrDefault(t, MyNewVar, "default-value")
}
```

For training images that support operator-injected defaults, use the three-level resolution in `defaults.go`:
1. Test env var (e.g., `TEST_TRAINING_CUDA_PYTORCH_28_IMAGE`)
2. Operator `RELATED_IMAGE_*` env var
3. Hardcoded default in `defaults.go`

## Writing unit tests

Use `NewTest(t)` from `fakeclient.go` to create a test context with fake clients:

```go
func TestMyHelper(t *testing.T) {
    test := NewTest(t)

    // Create test fixtures via fake client
    resource := &corev1.Pod{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "test-pod",
            Namespace: "test-namespace",
        },
    }
    test.client.Core().CoreV1().Pods("test-namespace").Create(test.ctx, resource, metav1.CreateOptions{})

    // Call the function under test
    result := GetPods(test, "test-namespace", metav1.ListOptions{})

    // Assert
    test.Expect(result).Should(gomega.HaveLen(1))
    test.Expect(result[0].Name).To(gomega.Equal("test-pod"))
}
```

See `core_test.go`, `trainjob_test.go`, `environment_test.go` for more examples.

## Per-suite extensions

Put helpers in per-suite `support.go` (e.g., `tests/trainer/support.go`) when they:
- Use embedded test resources specific to that suite
- Reference suite-specific APIs or configurations
- Would not be useful to other test suites

Put helpers in `tests/common/support/` when they:
- Work with standard Kubernetes or shared custom resources
- Could be reused across multiple test suites

## Validation

```bash
make unit-test                                        # Run all support lib unit tests
make golangci-lint LINT_PKG=./tests/common/support/...  # Lint the support package
go vet ./tests/common/support/...                     # Vet the support package
make verify-imports                                   # Verify import ordering
```

## Checklist

- [ ] New helpers follow the async getter or resource creation pattern
- [ ] `GenerateName` used for all created resources
- [ ] `t.T().Helper()` called at the top of every helper function
- [ ] Unit tests added in a corresponding `_test.go` file
- [ ] `make unit-test` passes
- [ ] `make golangci-lint LINT_PKG=./tests/common/support/...` passes
- [ ] `make verify-imports` passes
