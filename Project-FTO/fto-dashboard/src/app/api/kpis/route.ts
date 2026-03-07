import { NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

export async function GET() {
  try {
    const kpis = await prisma.kpiDefinition.findMany({
      orderBy: {
        sortOrder: 'asc'
      }
    })

    return NextResponse.json({
      kpis
    })
  } catch (error) {
    console.error('Error fetching KPIs:', error)
    return NextResponse.json(
      { error: 'Failed to fetch KPI definitions' },
      { status: 500 }
    )
  }
}
