/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package trainer

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"

	. "github.com/opendatahub-io/distributed-workloads/tests/common/support"
)

const (
	// ConstantBucketName is the bucket name used for all test cases
	// This ensures consistent bucket reuse across tests, even if cleanup fails
	ConstantBucketName = "training-kubeflow"
)

// CloudURI represents a parsed cloud storage URI
type CloudURI struct {
	Scheme string // e.g., "s3", "azure", "gs"
	Bucket string
	Prefix string
}

// ParseCloudURI parses a cloud storage URI into scheme, bucket, and prefix.
// Excludes PVC URIs (pvc://) and local filesystem paths (no scheme).
// Returns nil if not a valid cloud storage URI.
func ParseCloudURI(uri string) *CloudURI {
	if idx := strings.Index(uri, "://"); idx <= 0 {
		return nil // Local filesystem path (no scheme)
	}

	scheme := uri[:strings.Index(uri, "://")]
	// Exclude PVC URIs - they are not cloud storage
	if scheme == "pvc" {
		return nil
	}

	rest := uri[strings.Index(uri, "://")+3:]
	parts := strings.SplitN(rest, "/", 2)
	bucket := parts[0]
	prefix := ""
	if len(parts) > 1 {
		prefix = parts[1]
	}

	return &CloudURI{
		Scheme: scheme,
		Bucket: bucket,
		Prefix: prefix,
	}
}

// CloudStorageProvider defines the interface for cloud storage operations.
// Easy to extend: implement for Azure, GCS, etc.
type CloudStorageProvider interface {
	// CreateBucket creates a new bucket with the specified region
	CreateBucket(ctx context.Context, bucketName, region string) error
	// DeleteBucket deletes a bucket and all its contents
	DeleteBucket(ctx context.Context, bucketName string) error
	// BucketExists checks if a bucket exists
	BucketExists(ctx context.Context, bucketName string) (bool, error)
	// CheckpointExists verifies at least one checkpoint exists at the URI
	CheckpointExists(ctx context.Context, uri string) bool
}

// S3Provider implements CloudStorageProvider for S3-compatible storage
// Uses singleton pattern to reuse the same client across all operations
type S3Provider struct {
	client *minio.Client
}

var (
	s3ProviderOnce sync.Once
	s3Provider     *S3Provider
	s3ProviderErr  error
)

// GetS3Provider returns a singleton S3 provider with a single reused client.
// The client is created once on first call and reused for all subsequent calls.
// This is more efficient than creating a new client for each operation.
func GetS3Provider() (*S3Provider, error) {
	s3ProviderOnce.Do(func() {
		endpoint, _ := GetStorageBucketDefaultEndpoint()
		accessKey, _ := GetStorageBucketAccessKeyId()
		secretKey, _ := GetStorageBucketSecretKey()

		if endpoint == "" || accessKey == "" || secretKey == "" {
			s3ProviderErr = fmt.Errorf("S3 credentials not configured")
			return
		}

		// Parse endpoint to determine protocol (HTTP vs HTTPS)
		secure := !strings.HasPrefix(endpoint, "http://")
		endpoint = strings.TrimPrefix(strings.TrimPrefix(endpoint, "https://"), "http://")

		// Configure TLS transport to skip certificate verification
		// Why: S3-compatible storage (AWS S3, MinIO, Ceph, etc.) in test environments typically uses self-signed certificates
		// Without this: HTTPS requests fail with "x509: certificate signed by unknown authority"
		// Security: This is test-only code - production should use properly signed certificates
		transport := &http.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true, // Skip cert verification for test environments
			},
		}

		// Create client ONCE and reuse it
		client, err := minio.New(endpoint, &minio.Options{
			Creds:     credentials.NewStaticV4(accessKey, secretKey, ""),
			Secure:    secure,
			Transport: transport,
		})
		if err != nil {
			s3ProviderErr = fmt.Errorf("failed to create S3 client: %w", err)
			return
		}

		s3Provider = &S3Provider{client: client}
	})

	return s3Provider, s3ProviderErr
}

// CheckpointExists verifies at least one checkpoint exists at the S3 URI.
// URI format: s3://bucket/prefix/path
// Returns false if checkpoints don't exist OR if any error occurs (can't connect, invalid URI, etc.)
// In test context, both cases should fail the test anyway.
func (p *S3Provider) CheckpointExists(ctx context.Context, uri string) bool {
	// Parse URI: s3://bucket/prefix -> bucket, prefix
	parts := strings.TrimPrefix(uri, "s3://")
	idx := strings.Index(parts, "/")
	if idx < 0 {
		return false // Invalid URI format
	}
	bucket := parts[:idx]
	prefix := parts[idx+1:] + "/"

	// List objects under prefix
	objectsCh := p.client.ListObjects(ctx, bucket, minio.ListObjectsOptions{
		Prefix:    prefix,
		Recursive: true,
	})

	// Check if at least one valid checkpoint object exists
	for obj := range objectsCh {
		if obj.Err != nil {
			return false // Error listing objects
		}
		// Skip .incomplete markers - we only want complete checkpoints
		if !strings.Contains(obj.Key, ".incomplete") {
			return true // Found at least one checkpoint
		}
	}

	return false // No checkpoints found
}

// CreateBucket creates a new S3 bucket in the specified region
// Assumes bucket name is unique (e.g., with timestamp) and doesn't already exist
func (p *S3Provider) CreateBucket(ctx context.Context, bucketName, region string) error {
	if bucketName == "" {
		return fmt.Errorf("bucket name cannot be empty")
	}

	// Default to us-east-1 if no region specified
	if region == "" {
		region = "us-east-1"
	}

	// Create the bucket with region
	if err := p.client.MakeBucket(ctx, bucketName, minio.MakeBucketOptions{
		Region: region,
	}); err != nil {
		return fmt.Errorf("failed to create bucket %s in region %s: %w", bucketName, region, err)
	}

	return nil
}

// DeleteBucket deletes an S3 bucket and all its contents
func (p *S3Provider) DeleteBucket(ctx context.Context, bucketName string) error {
	if bucketName == "" {
		return fmt.Errorf("bucket name cannot be empty")
	}

	// Check if bucket exists
	exists, err := p.client.BucketExists(ctx, bucketName)
	if err != nil {
		return fmt.Errorf("failed to check if bucket exists: %w", err)
	}

	if !exists {
		return nil // Bucket doesn't exist, nothing to delete
	}

	// Delete all objects in the bucket first
	objectsCh := p.client.ListObjects(ctx, bucketName, minio.ListObjectsOptions{
		Recursive: true,
	})

	for object := range objectsCh {
		if object.Err != nil {
			return fmt.Errorf("failed to list objects: %w", object.Err)
		}
		if err := p.client.RemoveObject(ctx, bucketName, object.Key, minio.RemoveObjectOptions{}); err != nil {
			return fmt.Errorf("failed to delete object %s: %w", object.Key, err)
		}
	}

	// Delete the bucket
	if err := p.client.RemoveBucket(ctx, bucketName); err != nil {
		return fmt.Errorf("failed to delete bucket %s: %w", bucketName, err)
	}

	return nil
}

// BucketExists checks if an S3 bucket exists
func (p *S3Provider) BucketExists(ctx context.Context, bucketName string) (bool, error) {
	if bucketName == "" {
		return false, fmt.Errorf("bucket name cannot be empty")
	}

	exists, err := p.client.BucketExists(ctx, bucketName)
	if err != nil {
		return false, fmt.Errorf("failed to check if bucket exists: %w", err)
	}

	return exists, nil
}
