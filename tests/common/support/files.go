package support

import (
	"os"

	gomega "github.com/onsi/gomega"
)

func ReadFile(t Test, fileName string) []byte {
	t.T().Helper()
	file, err := os.ReadFile(fileName)
	t.Expect(err).NotTo(gomega.HaveOccurred())
	return file
}
