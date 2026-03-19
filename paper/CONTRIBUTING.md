# Guide de contribution pour l'article JOSS

## Contraintes de longueur JOSS

L'article JOSS doit respecter les contraintes suivantes :

- **Longueur totale** : 750-1750 mots (corps principal uniquement, hors frontmatter YAML et références)
- Les auteurs dont l'article dépasse significativement 1750 mots peuvent être invités à réduire la longueur
- Les articles "full length" ne sont pas permis
- La documentation API doit être dans la documentation du logiciel, pas dans l'article

### Vérification du nombre de mots

Pour vérifier le nombre de mots dans l'article :

```bash
# Compter les mots du corps principal (hors frontmatter et références)
sed -n '/^# Summary/,/^# References/p' paper.md | head -n -1 | wc -w
```

**Important** : Toujours vérifier le nombre de mots après chaque modification de `paper.md`.

## Sections obligatoires

L'article JOSS doit contenir les sections suivantes :

1. **Summary** : Résumé concis du logiciel
2. **Statement of Need** : Justification du besoin et de l'utilité du logiciel
3. **State of the Field** (optionnel mais recommandé) : Contexte et outils existants
4. **Software Design** : Architecture et composants principaux
5. **Research Impact Statement** (optionnel) : Impact sur la recherche
6. **AI Usage Disclosure** (si applicable) : Utilisation d'outils d'IA
7. **Acknowledgements** : Remerciements et financements
8. **References** : Références bibliographiques

## Compilation de l'article

Pour compiler l'article en PDF :

```bash
cd /home/chanial/work/scipol/furax/paper-joss
docker run --rm -v $(pwd):/data -u $(id -u):$(id -g) openjournals/inara -o pdf,crossref paper.md
```

Le PDF sera généré dans le même répertoire.

## Références

- [JOSS Paper Format](https://joss.readthedocs.io/en/latest/paper.html)
- [JOSS Submission Guidelines](https://joss.readthedocs.io/en/latest/submitting.html)
