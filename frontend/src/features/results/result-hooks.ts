import { useQuery } from "@tanstack/react-query"

import { getArtifactRows, type ArtifactReferences } from "../../api/client"

export function useAnalyticsArtifacts(jobId: string, artifacts: ArtifactReferences) {
  const classes = useQuery({
    queryKey: ["artifact", jobId, "class_distribution_csv"],
    queryFn: () => getArtifactRows<Record<string, string>>(jobId, "class_distribution_csv"),
    enabled: Boolean(artifacts.class_distribution_csv),
    retry: false,
  })
  const directions = useQuery({
    queryKey: ["artifact", jobId, "direction_distribution_csv"],
    queryFn: () => getArtifactRows<Record<string, string>>(jobId, "direction_distribution_csv"),
    enabled: Boolean(artifacts.direction_distribution_csv),
    retry: false,
  })
  const traffic = useQuery({
    queryKey: ["artifact", jobId, "traffic_over_time_csv"],
    queryFn: () => getArtifactRows<Record<string, string>>(jobId, "traffic_over_time_csv"),
    enabled: Boolean(artifacts.traffic_over_time_csv),
    retry: false,
  })
  return {
    classRows: classes.data ?? [],
    directionRows: directions.data ?? [],
    trafficRows: traffic.data ?? [],
    loading: classes.isLoading || directions.isLoading || traffic.isLoading,
    error: [classes.error, directions.error, traffic.error].find(Boolean) as Error | null | undefined,
  }
}
