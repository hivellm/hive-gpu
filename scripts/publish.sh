#!/bin/bash

# Hive-GPU Publishing Script
# This script helps publish hive-gpu to crates.io

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if version exists on crates.io
check_version_exists() {
    local version=$1
    print_status "Checking if version $version already exists on crates.io..."
    
    if cargo search hive-gpu --limit 1 | grep -q "hive-gpu = \"$version\""; then
        print_error "Version $version already exists on crates.io"
        return 1
    fi
    
    print_success "Version $version is available for publishing"
    return 0
}

# Function to update version in Cargo.toml
update_version() {
    local version=$1
    print_status "Updating Cargo.toml to version $version..."
    
    # Update version in Cargo.toml
    sed -i.bak "s/^version = .*/version = \"$version\"/" Cargo.toml
    
    # Verify the change
    if grep -q "version = \"$version\"" Cargo.toml; then
        print_success "Updated Cargo.toml to version $version"
    else
        print_error "Failed to update version in Cargo.toml"
        return 1
    fi
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Run basic tests
    cargo test --verbose
    
    # Run tests with different features
    print_status "Running tests with metal-native feature..."
    cargo test --features metal-native --verbose || print_warning "Metal tests skipped (not on macOS)"
    
    print_status "Running tests with cuda feature..."
    cargo test --features cuda --verbose || print_warning "CUDA tests skipped (not on Linux with CUDA)"
    
    print_status "Running tests with wgpu feature..."
    cargo test --features wgpu --verbose
    
    print_success "All tests passed!"
}

# Function to build and check
build_and_check() {
    print_status "Building with all features..."
    cargo build --all-features --verbose
    
    print_status "Checking package..."
    cargo check --all-features
    cargo package --verbose
    
    print_success "Build and check completed successfully!"
}

# Function to publish to crates.io
publish_to_crates() {
    print_status "Publishing to crates.io..."
    
    # Check if CARGO_REGISTRY_TOKEN is set
    if [ -z "$CARGO_REGISTRY_TOKEN" ]; then
        print_error "CARGO_REGISTRY_TOKEN environment variable is not set"
        print_status "Please set your crates.io token:"
        print_status "export CARGO_REGISTRY_TOKEN=your_token_here"
        return 1
    fi
    
    # Login to crates.io
    echo "$CARGO_REGISTRY_TOKEN" | cargo login
    
    # Publish
    cargo publish --verbose
    
    print_success "Successfully published to crates.io!"
}

# Function to verify publication
verify_publication() {
    local version=$1
    print_status "Verifying publication..."
    
    # Wait a bit for crates.io to update
    sleep 30
    
    # Check if the version is available
    if cargo search hive-gpu --limit 1 | grep -q "hive-gpu = \"$version\""; then
        print_success "Version $version is now available on crates.io!"
    else
        print_warning "Version $version might not be available yet. Please check manually."
    fi
}

# Function to create git tag
create_git_tag() {
    local version=$1
    print_status "Creating git tag v$version..."
    
    git tag "v$version"
    git push origin "v$version"
    
    print_success "Created and pushed tag v$version"
}

# Main function
main() {
    local version=$1
    
    if [ -z "$version" ]; then
        print_error "Usage: $0 <version>"
        print_status "Example: $0 0.1.0"
        exit 1
    fi
    
    # Remove 'v' prefix if present
    version=${version#v}
    
    print_status "Starting publication process for hive-gpu v$version"
    
    # Check if version exists
    if ! check_version_exists "$version"; then
        exit 1
    fi
    
    # Update version
    update_version "$version"
    
    # Run tests
    run_tests
    
    # Build and check
    build_and_check
    
    # Publish to crates.io
    publish_to_crates
    
    # Verify publication
    verify_publication "$version"
    
    # Create git tag
    create_git_tag "$version"
    
    print_success "🎉 Successfully published hive-gpu v$version!"
    print_status "📦 Crate: https://crates.io/crates/hive-gpu"
    print_status "📚 Docs: https://docs.rs/hive-gpu"
    print_status "🏷️  Tag: v$version"
}

# Run main function with all arguments
main "$@"
