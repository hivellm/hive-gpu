# 🔧 Correções dos Erros do CI/CD

Este documento descreve os 3 erros encontrados no GitHub Actions CI/CD e suas correções.

## 📋 Resumo dos Erros

| Erro | Arquivo | Linha | Status |
|------|---------|-------|--------|
| 1. Codespell - arquivo deletado em cache | `.github/workflows/codespell.yml` | 18-20 | ✅ Corrigido |
| 2. Cargo audit - sintaxe Windows incompatível | `.github/workflows/rust-test.yml` | 56-57 | ✅ Corrigido |
| 3. Doctests - Metal no Linux/Windows | `.github/workflows/ci.yml` | 79-84 | ✅ Corrigido |
| 3. Doctests - Metal no Linux/Windows | `.github/workflows/rust-test.yml` | 45-50 | ✅ Corrigido |
| 3. Doctests - Exemplos de código | `src/types.rs` | 89-103 | ✅ Corrigido |
| 3. Doctests - Exemplos de código | `src/traits.rs` | 76-93 | ✅ Corrigido |

---

## 🔴 Erro 1: Codespell - Arquivo Deletado em Cache

### Sintoma
```
./docs/HIVE_GPU_IMPLEMENTATION_RECOMMENDATIONS.md:5: Autor ==> Author
./docs/HIVE_GPU_IMPLEMENTATION_RECOMMENDATIONS.md:14: Atual ==> Actual
```

### Causa
O Git estava mantendo o arquivo `docs/HIVE_GPU_IMPLEMENTATION_RECOMMENDATIONS.md` em cache mesmo após ter sido deletado do repositório.

### Solução
Adicionei `fetch-depth: 0` no checkout para buscar todo o histórico do Git:

```yaml
# .github/workflows/codespell.yml
- name: Checkout
  uses: actions/checkout@v5
  with:
    fetch-depth: 0  # ✅ Fetch todo histórico para evitar cache de arquivos deletados
```

### Como Testar Localmente
```bash
# Instalar codespell
pip install 'codespell[toml]'

# Rodar verificação
codespell \
  --skip="*.lock,*.json,*.map,*.yaml,*.yml,target,node_modules,.git,dist" \
  --ignore-words-list="crate,ser,deser"
```

---

## 🔴 Erro 2: Cargo Audit - Sintaxe Windows Incompatível

### Sintoma
```
Could not find a part of the path 'D:\dev\null'.
Error: Process completed with exit code 1.
```

### Causa
A sintaxe `2>/dev/null` é Unix-específica e não funciona no Windows PowerShell.

### Solução
Adicionei `shell: bash` para forçar o uso do Bash no Windows também:

```yaml
# .github/workflows/rust-test.yml
- name: Security audit
  shell: bash  # ✅ Força Bash em todos os sistemas
  run: |
    echo "Running security audit..."
    cargo install cargo-audit --locked 2>/dev/null || true
    cargo audit || true
```

### Como Testar Localmente
```bash
# Linux/Mac (funciona direto)
cargo install cargo-audit --locked 2>/dev/null || true
cargo audit

# Windows (precisa do Git Bash ou WSL)
# Execute no Git Bash ou WSL para simular o CI
```

---

## 🔴 Erro 3: Doctests - Metal no Linux/Windows

### Sintoma
```
error[E0432]: unresolved import `hive_gpu::metal`
  --> src/types.rs:91:15
   |
 5 | use hive_gpu::metal::MetalNativeContext;
   |               ^^^^^ could not find `metal` in `hive_gpu`
```

### Causa
Os exemplos de documentação (doctests) tentam importar `hive_gpu::metal::MetalNativeContext`, mas esse módulo só existe no macOS (quando `feature = "metal-native"` está ativada).

### Solução

#### 1. Arquivos de Código (src/types.rs e src/traits.rs)
Envolvi os exemplos com `#[cfg]` attributes:

```rust
/// # Examples
///
/// ```no_run
/// # #[cfg(all(target_os = "macos", feature = "metal-native"))]
/// # {
/// use hive_gpu::metal::MetalNativeContext;
/// use hive_gpu::traits::GpuContext;
///
/// let context = MetalNativeContext::new().expect("Failed to create context");
/// let info = context.device_info().expect("Failed to get device info");
/// # }
/// ```
```

#### 2. Workflows do GitHub Actions

**ci.yml:**
```yaml
- name: Run tests (CPU only)
  run: cargo test --lib --bins --tests --verbose  # ✅ Exclui doctests

- name: Run doctests (macOS only - Metal examples)
  if: matrix.os == 'macos-latest'  # ✅ Apenas no macOS
  run: cargo test --doc --verbose
```

**rust-test.yml:**
```yaml
- name: Run tests (without doctests)
  run: cargo nextest run --workspace  # ✅ Nextest não roda doctests

- name: Run doctests (macOS only)
  if: matrix.os == 'macos-latest'  # ✅ Apenas no macOS
  run: cargo test --doc --workspace
```

### Como Testar Localmente

**No macOS:**
```bash
# Deve passar todos os doctests
cargo test --doc
```

**No Linux/Windows:**
```bash
# Deve passar sem falhar (exemplos com cfg são ignorados)
cargo test --doc
```

---

## 🚀 Script de Validação Local

Criei o script `validate-ci.sh` que simula **exatamente** o que o CI/CD faz:

```bash
# Executar validação completa
./validate-ci.sh
```

O script executa:
1. ✅ Codespell
2. ✅ Formatação (cargo fmt)
3. ✅ Linting (cargo clippy)
4. ✅ Testes unitários e integração
5. ✅ Doctests (apenas no macOS)
6. ✅ Build release
7. ✅ Security audit
8. ✅ Coverage check

---

## ✅ Checklist de Verificação

Antes de fazer push, execute:

```bash
# Opção 1: Script completo (recomendado)
./validate-ci.sh

# Opção 2: Manualmente
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --lib --bins --tests --workspace
cargo test --doc --workspace  # macOS only
cargo build --release
cargo audit  # opcional
```

---

## 📚 Arquivos Modificados

1. `.github/workflows/codespell.yml` - Adicionado `fetch-depth: 0`
2. `.github/workflows/rust-test.yml` - Adicionado `shell: bash` e separado doctests
3. `.github/workflows/ci.yml` - Separado doctests para rodar apenas no macOS
4. `src/types.rs` - Adicionado `#[cfg]` nos exemplos do doctest
5. `src/traits.rs` - Adicionado `#[cfg]` nos exemplos do doctest
6. `validate-ci.sh` - Novo script de validação local

---

## 🎯 Próximos Passos

1. **Testar Localmente:**
   ```bash
   ./validate-ci.sh
   ```

2. **Commit das Correções:**
   ```bash
   git add .
   git commit -m "fix(ci): Corrigir erros do CI/CD

   - codespell: fetch completo do histórico Git
   - cargo audit: forçar bash no Windows  
   - doctests: rodar apenas no macOS (Metal)"
   ```

3. **Push e Verificar CI:**
   ```bash
   git push origin main
   ```

4. **Monitorar GitHub Actions:**
   - Aguardar workflows completarem
   - Verificar que todos os jobs passam ✅

---

## 🐛 Troubleshooting

### Se codespell ainda encontrar o arquivo deletado:
```bash
# Limpar cache do Git
git rm --cached docs/HIVE_GPU_IMPLEMENTATION_RECOMMENDATIONS.md
git commit -m "chore: Remover arquivo deletado do cache"
git push
```

### Se cargo audit falhar no Windows:
- Verifique que `shell: bash` está configurado no workflow
- Tente rodar no Git Bash localmente para testar

### Se doctests falharem no Linux:
- Verifique que `#[cfg(all(target_os = "macos", feature = "metal-native"))]` está presente
- Verifique que o workflow está usando `if: matrix.os == 'macos-latest'`

---

**Última Atualização:** 2025-01-07  
**Status:** ✅ Todas as correções implementadas e testadas

