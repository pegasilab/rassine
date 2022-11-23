@test "pyproject.toml version matches most recent Git tag" {
    PROJECT_VERSION=$(poetry version --short)
    TAG=$(git describe HEAD --tags --abbrev=0)
    if [[ "$TAG" != "$PROJECT_VERSION" ]]; then
        echo "Project version is $PROJECT_VERSION while latest Git tag is $TAG"
        exit 1
    fi
}
