@test "pyproject.toml version matches most recent Git tag" {
    PROJECT_VERSION=$(poetry version --short)
    TAG=$(git describe HEAD --tags --abbrev=0)
    if [[ "$TAG" != "v$PROJECT_VERSION" ]]; then
        echo "Project version is $PROJECT_VERSION while latest Git tag is $TAG"
        echo "The tag has format v[major].[minor].[patch]"
        exit 1
    fi
}
