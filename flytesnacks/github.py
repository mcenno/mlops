"""Github helpers."""


def create_pr(
    repo: str,
    branch_name: str,
    pr_title: str,
    pr_description: str,
    files: dict,
    token: str,
    source_branch: str = "main",
) -> None:
    """
    Create a branch `branch_name` in a github repositry `repo`, commit `files`, and open a PR.

    Args:
        repo (str): repository to create the PR in. Format: `<owner or organization>/<repo>`
        branch_name (str): name of the branch that is supposed to be created
        pr_title (str): Title of the PR
        pr_description (str): Description of the PR
        files (dict[str, str]): a dictionary containing filepaths and the contents of
            the respective files as key-value pairs.
        token (str): Github access token with repo permissions for authentication
    Kwargs:
        source_branch (str): branch to merge the feature branch into. Defaults to "main"
    """
    from github import Github

    g = Github(token)
    repo = g.get_repo(repo)

    if branch_name not in [b.name for b in list(repo.get_branches())]:
        main_branch = repo.get_branch(source_branch)
        repo.create_git_ref(ref="refs/heads/" + branch_name, sha=main_branch.commit.sha)

    for path, content in files.items():
        repo.create_file(
            path,
            f"Add {path}",
            content,
            branch_name,
        )

    repo.create_pull(title=pr_title, body=pr_description, head=branch_name, base=source_branch)
