# 🔐 GitHub Secrets Configuration

## Required Secrets for CI/CD

### 1. CARGO_REGISTRY_TOKEN
**Purpose**: Authentication token for publishing to crates.io

**How to get**:
1. Go to [crates.io](https://crates.io)
2. Login with your GitHub account
3. Go to Account Settings → API Tokens
4. Create a new token with name "GitHub Actions"
5. Copy the token

**How to set**:
1. Go to your GitHub repository
2. Go to Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `CARGO_REGISTRY_TOKEN`
5. Value: Your crates.io token

### 2. GITHUB_TOKEN (Automatic)
**Purpose**: GitHub API access for creating releases

**Note**: This is automatically provided by GitHub Actions, no manual setup required.

## Optional Secrets

### 3. CODECOV_TOKEN (Optional)
**Purpose**: Code coverage reporting

**How to get**:
1. Go to [codecov.io](https://codecov.io)
2. Login with GitHub
3. Add your repository
4. Get the token from repository settings

### 4. RUST_LOG (Optional)
**Purpose**: Logging level for debugging

**Value**: `debug` or `info`

## Environment Variables

### For Local Development
```bash
# Set your crates.io token
export CARGO_REGISTRY_TOKEN=your_token_here

# Set logging level
export RUST_LOG=debug
```

### For GitHub Actions
These are automatically set by the workflow:
- `CARGO_TERM_COLOR=always`
- `RUST_BACKTRACE=1`

## Verification

### Check if secrets are set:
1. Go to your repository
2. Go to Settings → Secrets and variables → Actions
3. Verify that `CARGO_REGISTRY_TOKEN` is listed

### Test locally:
```bash
# Test crates.io login
echo $CARGO_REGISTRY_TOKEN | cargo login

# Test publishing (dry run)
cargo publish --dry-run
```

## Security Notes

- Never commit tokens to the repository
- Use repository secrets for sensitive data
- Rotate tokens regularly
- Use least-privilege access
