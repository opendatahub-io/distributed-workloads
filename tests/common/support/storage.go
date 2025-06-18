package support

import (
	"fmt"

	"github.com/onsi/gomega"

	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var rwxSupportedProvisioners = map[string]bool{
	"nfs.csi.k8s.io": true,
}

func GetStorageClasses(t Test) []storagev1.StorageClass {
	t.T().Helper()
	scList, err := t.Client().Storage().StorageClasses().List(t.Ctx(), metav1.ListOptions{})
	t.Expect(err).NotTo(gomega.HaveOccurred())
	return scList.Items
}

func GetRWXStorageClass(t Test) (*storagev1.StorageClass, error) {
	t.T().Helper()

	storageClasses := GetStorageClasses(t)

	for _, sc := range storageClasses {
		if rwxSupportedProvisioners[sc.Provisioner] {
			t.T().Logf("Found StorageClass '%s' with provisioner '%s' which is likely to support RWX.", sc.Name, sc.Provisioner)
			return &sc, nil
		}
	}

	return nil, fmt.Errorf("no StorageClass found that is known to support ReadWriteMany (RWX) access mode")
}
