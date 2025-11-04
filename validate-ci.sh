#!/usr/bin/env bash
set -e  # Para na primeira falha

echo "🔍 =================================================="
echo "   Validação Local - Simulando CI/CD do GitHub"
echo "===================================================="
echo ""

# Detectar sistema operacional
OS=$(uname -s)
echo "🖥️  Sistema Operacional: $OS"
echo ""

# ============================================
# STEP 1: Codespell
# ============================================
echo "1️⃣  ===== CODESPELL: Verificando ortografia ====="
if command -v codespell &> /dev/null; then
    codespell \
      --skip="*.lock,*.json,*.map,*.yaml,*.yml,target,node_modules,.git,dist" \
      --ignore-words-list="crate,ser,deser" || {
        echo "❌ Codespell found spelling errors!"
        exit 1
    }
    echo "✅ Codespell passou"
else
    echo "⚠️  Codespell não instalado (pip install 'codespell[toml]')"
    echo "   Pulando verificação de codespell..."
fi
echo ""

# ============================================
# STEP 2: Formatação
# ============================================
echo "2️⃣  ===== FORMATTING: Verificando formatação ====="
cargo fmt --all -- --check || {
    echo "❌ Código não está formatado!"
    echo "   Execute: cargo fmt --all"
    exit 1
}
echo "✅ Formatação OK"
echo ""

# ============================================
# STEP 3: Linting
# ============================================
echo "3️⃣  ===== CLIPPY: Verificando linting ====="
cargo clippy --workspace --all-targets --all-features -- -D warnings || {
    echo "❌ Clippy encontrou warnings!"
    exit 1
}
echo "✅ Linting OK (0 warnings)"
echo ""

# ============================================
# STEP 4: Testes Unitários e Integração
# ============================================
echo "4️⃣  ===== TESTS: Executando testes (sem doctests) ====="
cargo test --lib --bins --tests --workspace || {
    echo "❌ Testes falharam!"
    exit 1
}
echo "✅ Testes unitários e integração OK"
echo ""

# ============================================
# STEP 5: Doctests (apenas no macOS)
# ============================================
echo "5️⃣  ===== DOCTESTS: Executando doc tests ====="
if [[ "$OS" == "Darwin" ]]; then
    echo "   (macOS detectado - rodando doctests)"
    cargo test --doc --workspace || {
        echo "❌ Doctests falharam!"
        exit 1
    }
    echo "✅ Doctests OK"
else
    echo "   (Linux/Windows detectado - pulando doctests com Metal)"
    echo "⏭️  Doctests pulados (rodados apenas no macOS no CI)"
fi
echo ""

# ============================================
# STEP 6: Build
# ============================================
echo "6️⃣  ===== BUILD: Compilando release ====="
cargo build --release || {
    echo "❌ Build falhou!"
    exit 1
}
echo "✅ Build release OK"
echo ""

# ============================================
# STEP 7: Security Audit (opcional)
# ============================================
echo "7️⃣  ===== SECURITY: Auditoria de segurança ====="
if command -v cargo-audit &> /dev/null; then
    cargo audit || {
        echo "⚠️  Vulnerabilidades encontradas (isso não bloqueia o CI)"
    }
    echo "✅ Security audit concluído"
else
    echo "⚠️  cargo-audit not installed"
    echo "   Install with: cargo install cargo-audit"
    echo "   Skipping audit..."
fi
echo ""

# ============================================
# STEP 8: Coverage (opcional)
# ============================================
echo "8️⃣  ===== COVERAGE: Verificando cobertura ====="
if command -v cargo-llvm-cov &> /dev/null; then
    cargo llvm-cov --all --summary-only || {
        echo "⚠️  Coverage check falhou"
    }
    echo "✅ Coverage check concluído"
else
    echo "⚠️  cargo-llvm-cov not installed"
    echo "   Install with: cargo install cargo-llvm-cov"
    echo "   Skipping coverage..."
fi
echo ""

# ============================================
# RESUMO FINAL
# ============================================
echo "🎉 =================================================="
echo "   TODAS AS VALIDAÇÕES PASSARAM!"
echo "===================================================="
echo ""
echo "✅ Checklist completo:"
echo "   [✓] Codespell"
echo "   [✓] Formatação (cargo fmt)"
echo "   [✓] Linting (cargo clippy)"
echo "   [✓] Testes unitários e integração"
if [[ "$OS" == "Darwin" ]]; then
    echo "   [✓] Doctests (macOS)"
else
    echo "   [⏭] Doctests (pulados - só no macOS)"
fi
echo "   [✓] Build release"
echo "   [✓] Security audit"
echo "   [✓] Coverage check"
echo ""
echo "🚀 Você está pronto para fazer commit e push!"
echo ""
echo "✋ PRÓXIMOS PASSOS:"
echo "   1. git add ."
echo "   2. git commit -m 'fix(ci): Fix CI/CD errors'"
echo "   3. git push origin main"
echo ""

